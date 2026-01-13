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
from typing import TYPE_CHECKING, Optional

from aragora.audience.suggestions import cluster_suggestions, format_for_prompt
from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.agent_pool import AgentPool, AgentPoolConfig
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.arena_phases import init_phases
from aragora.debate.audience_manager import AudienceManager
from aragora.debate.autonomic_executor import AutonomicExecutor
from aragora.debate.chaos_theater import DramaLevel, get_chaos_director
from aragora.debate.complexity_governor import (
    classify_task_complexity,
    get_complexity_governor,
)
from aragora.debate.context import DebateContext
from aragora.debate.convergence import ConvergenceDetector
from aragora.debate.event_bridge import EventEmitterBridge
from aragora.debate.event_bus import EventBus
from aragora.debate.extensions import ArenaExtensions
from aragora.debate.immune_system import get_immune_system
from aragora.debate.judge_selector import JudgeSelector
from aragora.debate.optional_imports import OptionalImports
from aragora.debate.phase_executor import PhaseExecutor
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.debate.result_formatter import ResultFormatter
from aragora.debate.roles_manager import RolesManager
from aragora.debate.safety import resolve_auto_evolve, resolve_prompt_evolution
from aragora.debate.sanitization import OutputSanitizer
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
        event_emitter=None,  # Optional event emitter for subscribing to user events
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
        enable_performance_monitor: bool = False,  # Auto-create PerformanceMonitor if True
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

        # Initialize phase classes for orchestrator decomposition
        self._init_phases()

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
            memory=config.memory,
            event_hooks=config.event_hooks,
            event_emitter=config.event_emitter,
            spectator=config.spectator,
            debate_embeddings=config.debate_embeddings,
            insight_store=config.insight_store,
            recorder=config.recorder,
            agent_weights=config.agent_weights,
            position_tracker=config.position_tracker,
            position_ledger=config.position_ledger,
            enable_position_ledger=config.enable_position_ledger,
            elo_system=config.elo_system,
            persona_manager=config.persona_manager,
            dissent_retriever=config.dissent_retriever,
            consensus_memory=config.consensus_memory,
            flip_detector=config.flip_detector,
            calibration_tracker=config.calibration_tracker,
            continuum_memory=config.continuum_memory,
            relationship_tracker=config.relationship_tracker,
            moment_detector=config.moment_detector,
            tier_analytics_tracker=config.tier_analytics_tracker,
            loop_id=config.loop_id,
            strict_loop_scoping=config.strict_loop_scoping,
            circuit_breaker=config.circuit_breaker,
            initial_messages=config.initial_messages,
            trending_topic=config.trending_topic,
            pulse_manager=config.pulse_manager,
            auto_fetch_trending=config.auto_fetch_trending,
            population_manager=config.population_manager,
            auto_evolve=config.auto_evolve,
            breeding_threshold=config.breeding_threshold,
            evidence_collector=config.evidence_collector,
            breakpoint_manager=config.breakpoint_manager,
            performance_monitor=config.performance_monitor,
            enable_performance_monitor=config.enable_performance_monitor,
            enable_telemetry=config.enable_telemetry,
            use_airlock=config.use_airlock,
            airlock_config=config.airlock_config,
            agent_selector=config.agent_selector,
            use_performance_selection=config.use_performance_selection,
            prompt_evolver=config.prompt_evolver,
            enable_prompt_evolution=config.enable_prompt_evolution,
            checkpoint_manager=config.checkpoint_manager,
            enable_checkpointing=config.enable_checkpointing,
            org_id=config.org_id,
            user_id=config.user_id,
            usage_tracker=config.usage_tracker,
            broadcast_pipeline=config.broadcast_pipeline,
            auto_broadcast=config.auto_broadcast,
            broadcast_min_confidence=config.broadcast_min_confidence,
            training_exporter=config.training_exporter,
            auto_export_training=config.auto_export_training,
            training_export_min_confidence=config.training_export_min_confidence,
        )

    def _init_core(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol | None,
        memory,
        event_hooks: dict | None,
        event_emitter,
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
            self.agents = wrap_agents(self.agents, airlock_cfg)
            logger.debug(f"[airlock] Wrapped {len(self.agents)} agents with resilience layer")
        self.memory = memory
        self.hooks = event_hooks or {}
        self.event_emitter = event_emitter
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

    def _broadcast_health_event(self, event: dict) -> None:
        """Broadcast health events to WebSocket clients via EventBus."""
        try:
            if self.event_bus:
                self.event_bus.emit_sync(
                    event_type="health_event",
                    debate_id=getattr(self, "_current_debate_id", ""),
                    **event.get("data", event),
                )
            else:
                # Fallback to event_bridge if event_bus not initialized yet
                self.event_bridge.notify(
                    event_type="health_event",
                    **event.get("data", event),
                )
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

    def _auto_init_position_ledger(self) -> None:
        """Auto-initialize PositionLedger for tracking agent positions.

        PositionLedger tracks every position agents take across debates,
        including outcomes and reversals. This enables:
        - Position accuracy tracking per agent
        - Reversal detection for flip analysis
        - Historical position queries for grounded personas
        """
        try:
            from aragora.agents.positions import PositionLedger

            self.position_ledger = PositionLedger()
            logger.debug("Auto-initialized PositionLedger for position tracking")
        except ImportError:
            logger.warning("PositionLedger not available - position tracking disabled")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("PositionLedger auto-init failed: %s", e)

    def _auto_init_calibration_tracker(self) -> None:
        """Auto-initialize CalibrationTracker when enable_calibration is True."""
        try:
            from aragora.agents.calibration import CalibrationTracker

            self.calibration_tracker = CalibrationTracker()
            logger.debug("Auto-initialized CalibrationTracker for prediction calibration")
        except ImportError:
            logger.warning("CalibrationTracker not available - calibration disabled")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("CalibrationTracker auto-init failed: %s", e)

    def _auto_init_dissent_retriever(self) -> None:
        """Auto-initialize DissentRetriever when consensus_memory is available.

        The DissentRetriever enables seeding new debates with historical minority
        views, helping agents avoid past groupthink and consider diverse perspectives.
        """
        try:
            from aragora.memory.consensus import DissentRetriever

            self.dissent_retriever = DissentRetriever(self.consensus_memory)
            logger.debug("Auto-initialized DissentRetriever for historical minority views")
        except ImportError:
            logger.debug("DissentRetriever not available - historical dissent disabled")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("DissentRetriever auto-init failed: %s", e)

    def _auto_init_moment_detector(self) -> None:
        """Auto-initialize MomentDetector when elo_system is available."""
        try:
            from aragora.agents.grounded import MomentDetector as MD

            self.moment_detector = MD(
                elo_system=self.elo_system,
                position_ledger=self.position_ledger,
                relationship_tracker=self.relationship_tracker,
            )
            logger.debug("Auto-initialized MomentDetector for significant moment detection")
        except ImportError:
            pass  # MomentDetector not available
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("MomentDetector auto-init failed: %s", e)

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

    def _init_convergence(self) -> None:
        """Initialize convergence detection if enabled."""
        self.convergence_detector = None
        if self.protocol.convergence_detection:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=self.protocol.convergence_threshold,
                divergence_threshold=self.protocol.divergence_threshold,
                min_rounds_before_check=1,
            )

        # Track responses for convergence detection
        self._previous_round_responses: dict[str, str] = {}

    def _init_caches(self) -> None:
        """Initialize caches for computed values."""
        # Cache for historical context (computed once per debate)
        self._historical_context_cache: str = ""

        # Cache for research context (computed once per debate)
        self._research_context_cache: Optional[str] = None

        # Cache for evidence pack (for grounding verdict with citations)
        self._research_evidence_pack = None

        # Cache for continuum memory context (retrieved once per debate)
        self._continuum_context_cache: str = ""
        self._continuum_retrieved_ids: list = []
        self._continuum_retrieved_tiers: dict = {}  # ID -> MemoryTier for analytics

        # Cached similarity backend for vote grouping (avoids recreating per call)
        self._similarity_backend = None

        # Cache for debate domain (computed once per debate)
        self._debate_domain_cache: Optional[str] = None

    def _init_phases(self) -> None:
        """Initialize phase classes for orchestrator decomposition."""
        init_phases(self)

        # Initialize PhaseExecutor with all phases for centralized execution
        from aragora.debate.phase_executor import PhaseConfig

        timeout = getattr(self.protocol, "timeout", 300.0)

        # Create wrapper for context_initializer to match Phase protocol
        # (context_initializer uses .initialize() instead of .execute())
        class ContextInitWrapper:
            name = "context_initializer"

            def __init__(self, initializer):
                self._initializer = initializer

            async def execute(self, context):
                return await self._initializer.initialize(context)

        self.phase_executor = PhaseExecutor(
            phases={
                "context_initializer": ContextInitWrapper(self.context_initializer),
                "proposal": self.proposal_phase,
                "debate_rounds": self.debate_rounds_phase,
                "consensus": self.consensus_phase,
                "analytics": self.analytics_phase,
                "feedback": self.feedback_phase,
            },
            config=PhaseConfig(
                total_timeout_seconds=timeout,
                phase_timeout_seconds=timeout / 3,  # Per-phase timeout
                enable_tracing=True,
            ),
        )

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
        self.prompt_builder._historical_context_cache = self._historical_context_cache
        self.prompt_builder._continuum_context_cache = self._get_continuum_context()
        self.prompt_builder.user_suggestions = self.user_suggestions  # type: ignore[assignment]

    def _get_continuum_context(self) -> str:
        """Retrieve relevant memories from ContinuumMemory for debate context.

        Delegates to ContextGatherer.get_continuum_context().
        """
        if self._continuum_context_cache:
            return self._continuum_context_cache

        if not self.continuum_memory:
            return ""

        domain = self._extract_debate_domain()
        context, retrieved_ids, retrieved_tiers = self.context_gatherer.get_continuum_context(
            continuum_memory=self.continuum_memory,
            domain=domain,
            task=self.env.task,
        )

        # Track retrieved IDs and tiers for outcome updates
        self._continuum_retrieved_ids = retrieved_ids
        self._continuum_retrieved_tiers = retrieved_tiers
        self._continuum_context_cache = context
        return context

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
            self._continuum_retrieved_ids,
            tiers=self._continuum_retrieved_tiers,
        )
        self.memory_manager.update_memory_outcomes(result)
        # Clear local tracking
        self._continuum_retrieved_ids = []
        self._continuum_retrieved_tiers = {}

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
                # Log high-priority citation needs
                high_priority = [n for n in needs if n["priority"] == "high"]
                if high_priority:
                    logger.debug(f"citations_needed agent={agent_name} count={len(high_priority)}")

        return citation_needs

    def _extract_debate_domain(self) -> str:
        """Extract domain from the debate task for calibration tracking.

        Uses heuristics to categorize the debate topic.
        Result is cached at both instance level (for this debate) and
        module level (for repeated tasks across debates).
        """
        # Return instance-level cached domain if available
        if self._debate_domain_cache is not None:
            return self._debate_domain_cache

        # Use module-level LRU cache for the actual computation
        domain = _compute_domain_from_task(self.env.task.lower())

        # Cache at instance level and return
        self._debate_domain_cache = domain
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

    def _record_grounded_position(
        self,
        agent_name: str,
        content: str,
        debate_id: str,
        round_num: int,
        confidence: float = 0.7,
        domain: Optional[str] = None,
    ):
        """Record a position to the grounded persona ledger."""
        if not self.position_ledger:
            return
        try:
            self.position_ledger.record_position(
                agent_name=agent_name,
                claim=content[:1000],
                confidence=confidence,
                debate_id=debate_id,
                round_num=round_num,
                domain=domain,
            )
        except (AttributeError, TypeError, ValueError) as e:
            # Expected parameter or state errors
            logger.warning(f"Position ledger error: {e}")
        except (KeyError, RuntimeError, OSError) as e:
            # Unexpected error - log type for debugging
            logger.warning(f"Position ledger error (type={type(e).__name__}): {e}")

    def _update_agent_relationships(
        self, debate_id: str, participants: list[str], winner: Optional[str], votes: list
    ):
        """Update agent relationships after debate completion.

        Uses batch update for O(1) database connections instead of O(n²) for n participants.
        """
        if not self.elo_system:
            return
        try:
            vote_choices = {
                v.agent: v.choice for v in votes if hasattr(v, "agent") and hasattr(v, "choice")
            }
            # Build batch of relationship updates
            updates = []
            for i, agent_a in enumerate(participants):
                for agent_b in participants[i + 1 :]:
                    agreed = (
                        agent_a in vote_choices
                        and agent_b in vote_choices
                        and vote_choices[agent_a] == vote_choices[agent_b]
                    )
                    a_win = 1 if winner == agent_a else 0
                    b_win = 1 if winner == agent_b else 0
                    updates.append(
                        {
                            "agent_a": agent_a,
                            "agent_b": agent_b,
                            "debate_increment": 1,
                            "agreement_increment": 1 if agreed else 0,
                            "a_win": a_win,
                            "b_win": b_win,
                        }
                    )
            # Single transaction for all updates
            self.elo_system.update_relationships_batch(updates)
        except (AttributeError, TypeError, KeyError) as e:
            # Expected data access errors
            logger.warning(f"Relationship update error: {e}")
        except (ValueError, RuntimeError, OSError) as e:
            # Unexpected error - log type for debugging
            logger.warning(f"Relationship update error (type={type(e).__name__}): {e}")

    def _create_grounded_verdict(self, result: "DebateResult"):
        """Create a GroundedVerdict for the final answer.

        Heavy3-inspired: Wrap final answers with evidence grounding analysis.
        Delegates to EvidenceGrounder for the actual grounding logic.
        """
        if not result.final_answer:
            return None

        return self.evidence_grounder.create_grounded_verdict(
            final_answer=result.final_answer,
            confidence=result.confidence,
        )

    async def _verify_claims_formally(self, result: "DebateResult") -> None:
        """Verify decidable claims using Z3 SMT solver.

        For arithmetic, logic, and constraint claims, attempts formal verification
        to provide machine-verified evidence. Delegates to EvidenceGrounder.
        """
        if not result.grounded_verdict:
            return

        await self.evidence_grounder.verify_claims_formally(result.grounded_verdict)

    async def _fetch_historical_context(self, task: str, limit: int = 3) -> str:
        """Fetch similar past debates for historical context."""
        return await self.memory_manager.fetch_historical_context(task, limit)

    def _format_patterns_for_prompt(self, patterns: list[dict]) -> str:
        """Format learned patterns as prompt context for agents."""
        return self.memory_manager._format_patterns_for_prompt(patterns)

    def _get_successful_patterns_from_memory(self, limit: int = 5) -> str:
        """Retrieve successful patterns from CritiqueStore memory."""
        return self.memory_manager.get_successful_patterns(limit)

    async def _perform_research(self, task: str) -> str:
        """Perform multi-source research for the debate topic and return formatted context.

        Delegates to ContextGatherer which uses EvidenceCollector with multiple connectors:
        - WebConnector: DuckDuckGo search for general web results
        - GitHubConnector: Code/docs from GitHub repositories
        - LocalDocsConnector: Local documentation files

        Also includes pulse/trending context when available.
        """
        result = await self.context_gatherer.gather_all(task)
        # Update local cache and evidence grounder for backwards compatibility
        self._research_evidence_pack = self.context_gatherer.evidence_pack
        self.evidence_grounder.set_evidence_pack(self.context_gatherer.evidence_pack)
        return result

    async def _gather_aragora_context(self, task: str) -> Optional[str]:
        """Gather Aragora-specific documentation context if relevant to task.

        Delegates to ContextGatherer.gather_aragora_context().
        """
        return await self.context_gatherer.gather_aragora_context(task)

    async def _gather_evidence_context(self, task: str) -> Optional[str]:
        """Gather evidence from web, GitHub, and local docs connectors.

        Delegates to ContextGatherer.gather_evidence_context().
        """
        result = await self.context_gatherer.gather_evidence_context(task)
        # Update local cache and evidence grounder for backwards compatibility
        self._research_evidence_pack = self.context_gatherer.evidence_pack
        self.evidence_grounder.set_evidence_pack(self.context_gatherer.evidence_pack)
        return result

    async def _gather_trending_context(self) -> Optional[str]:
        """Gather pulse/trending context from social platforms.

        Delegates to ContextGatherer.gather_trending_context().
        """
        return await self.context_gatherer.gather_trending_context()

    async def _refresh_evidence_for_round(
        self, combined_text: str, ctx: "DebateContext", round_num: int
    ) -> int:
        """Refresh evidence based on claims made during a debate round.

        Delegates to ContextGatherer.refresh_evidence_for_round().

        Args:
            combined_text: Combined text from proposals and critiques
            ctx: The DebateContext
            round_num: Current round number

        Returns:
            Number of new evidence snippets added
        """
        count, updated_pack = await self.context_gatherer.refresh_evidence_for_round(
            combined_text=combined_text,
            evidence_collector=self.evidence_collector,
            task=self.env.task if self.env else "",
            evidence_store_callback=self._store_evidence_in_memory,
        )

        if updated_pack and hasattr(self, "prompt_builder") and self.prompt_builder:
            self._research_evidence_pack = updated_pack
            self.evidence_grounder.set_evidence_pack(updated_pack)
            self.prompt_builder.set_evidence_pack(updated_pack)

        return count

    def _format_conclusion(self, result: "DebateResult") -> str:
        """Format a clear, readable debate conclusion with full context.

        Delegates to ResultFormatter for the actual formatting.
        """
        formatter = ResultFormatter()
        return formatter.format_conclusion(result)

    def _assign_roles(self):
        """Assign roles to agents based on protocol. Delegates to RolesManager."""
        self.roles_manager.assign_initial_roles()

    def _apply_agreement_intensity(self):
        """Apply agreement intensity guidance. Delegates to RolesManager."""
        self.roles_manager.apply_agreement_intensity()

    def _assign_stances(self, round_num: int = 0):
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

    async def _cleanup(self) -> None:
        """Internal cleanup implementation.

        Can be called directly if not using context manager.
        """
        # Cancel any pending arena-related asyncio tasks
        try:
            for task in asyncio.all_tasks():
                task_name = task.get_name() if hasattr(task, "get_name") else ""
                if task_name and task_name.startswith(("arena_", "debate_")):
                    task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=1.0,
                        )
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
        except Exception as e:
            logger.debug(f"Error cancelling tasks during cleanup: {e}")

        # Clear internal caches
        self._historical_context_cache = ""
        self._research_context_cache = None

        # Close checkpoint manager if we created it
        if self.checkpoint_manager and hasattr(self.checkpoint_manager, "close"):
            try:
                close_result = self.checkpoint_manager.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
            except Exception as e:
                logger.debug(f"Error closing checkpoint manager: {e}")

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

                # Check for phase failures
                if not execution_result.success:
                    error_phases = [
                        p.phase_name for p in execution_result.phases
                        if p.status.value == "failed"
                    ]
                    if error_phases:
                        logger.warning(f"Phase failures: {error_phases}")

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

                # Track circuit breaker state
                if self.circuit_breaker:
                    open_count = sum(
                        1
                        for state in getattr(self.circuit_breaker, "_agent_states", {}).values()
                        if getattr(state, "is_open", False)
                    )
                    track_circuit_breaker_state(open_count)

        # Trigger extensions (billing, training export)
        # Extensions handle their own error handling and won't fail the debate
        self.extensions.on_debate_complete(ctx, ctx.result, self.agents)

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

    def _format_successful_patterns(self, limit: int = 3) -> str:
        """Format successful critique patterns for prompt injection."""
        if not self.memory:
            return ""
        try:
            patterns = self.memory.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            for p in patterns:
                issue_preview = (
                    p.issue_text[:100] + "..." if len(p.issue_text) > 100 else p.issue_text
                )
                fix_preview = (
                    p.suggestion_text[:80] + "..."
                    if len(p.suggestion_text) > 80
                    else p.suggestion_text
                )
                lines.append(f"- **{p.issue_type}**: {issue_preview}")
                if fix_preview:
                    lines.append(f"  Fix: {fix_preview} ({p.success_count} successes)")
            return "\n".join(lines)
        except (AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError) as e:
            logger.warning(f"Pattern retrieval error: {e}")
            return ""

    def _update_role_assignments(self, round_num: int) -> None:
        """Update cognitive role assignments for the current round.

        Delegates to RolesManager for centralized role management.
        """
        debate_domain = self._extract_debate_domain()
        self.roles_manager.update_role_assignments(round_num, debate_domain)

        # Sync role assignments back to orchestrator for backward compatibility
        self.current_role_assignments = self.roles_manager.current_role_assignments

        # Log role assignments
        if self.current_role_assignments:
            roles_str = ", ".join(
                f"{name}: {assign.role.value}"
                for name, assign in self.current_role_assignments.items()
            )
            logger.debug(f"role_assignments round={round_num} roles={roles_str}")

    def _get_role_context(self, agent: Agent) -> str:
        """Get cognitive role context for an agent in the current round.

        Delegates to RolesManager for centralized role management.
        """
        return self.roles_manager.get_role_context(agent)

    def _get_persona_context(self, agent: Agent) -> str:
        """Get persona context for agent specialization."""
        if not self.persona_manager:
            return ""

        # Try to get persona from database
        persona = self.persona_manager.get_persona(agent.name)
        if not persona:
            # Try default persona based on agent type (e.g., "claude_proposer" -> "claude")
            agent_type = agent.name.split("_")[0].lower()
            from aragora.agents.personas import DEFAULT_PERSONAS

            if agent_type in DEFAULT_PERSONAS:
                # DEFAULT_PERSONAS contains Persona objects directly
                persona = DEFAULT_PERSONAS[agent_type]
            else:
                return ""

        return persona.to_prompt_context()

    def _get_flip_context(self, agent: Agent) -> str:
        """Get flip/consistency context for agent self-awareness.

        This helps agents be aware of their position history and avoid
        unnecessary flip-flopping while still allowing genuine position changes.
        """
        if not self.flip_detector:
            return ""

        try:
            consistency = self.flip_detector.get_agent_consistency(agent.name)

            # Skip if no position history yet
            if consistency.total_positions == 0:
                return ""

            # Only inject context if there are notable flips
            if consistency.total_flips == 0:
                return ""

            # Build context based on flip patterns
            lines = ["## Position Consistency Note"]

            # Warn about contradictions specifically
            if consistency.contradictions > 0:
                lines.append(
                    f"You have {consistency.contradictions} prior position contradiction(s) on record. "
                    "Consider your stance carefully before arguing against positions you previously held."
                )

            # Note retractions
            if consistency.retractions > 0:
                lines.append(
                    f"You have retracted {consistency.retractions} previous position(s). "
                    "If changing positions again, clearly explain your reasoning."
                )

            # Add overall consistency score
            score = consistency.consistency_score
            if score < 0.7:
                lines.append(
                    f"Your consistency score is {score:.0%}. Prioritize coherent positions."
                )

            # Note domains with instability
            if consistency.domains_with_flips:
                domains = ", ".join(consistency.domains_with_flips[:3])
                lines.append(f"Domains with position changes: {domains}")

            return "\n".join(lines) if len(lines) > 1 else ""

        except (AttributeError, TypeError, ValueError, KeyError, RuntimeError) as e:
            logger.warning(f"Flip context error for {agent.name}: {e}")
            return ""

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
        # Drain pending audience events
        self._drain_user_events()

        # Sync state to PromptBuilder
        self._sync_prompt_builder_state()

        # Compute audience section if enabled and suggestions exist
        if not (
            self.protocol.audience_injection in ("summary", "inject") and self.user_suggestions
        ):
            return ""

        clusters = cluster_suggestions(list(self.user_suggestions))
        audience_section = format_for_prompt(clusters)

        # Emit stream event for dashboard if requested
        if emit_event and self.spectator and clusters:
            self._notify_spectator(
                "audience_summary",
                details=f"{sum(c.count for c in clusters)} suggestions in {len(clusters)} clusters",
                metric=len(clusters),
            )

        return audience_section

    def _build_proposal_prompt(self, agent: Agent) -> str:
        """Build the initial proposal prompt."""
        audience_section = self._prepare_audience_context(emit_event=True)
        return self.prompt_builder.build_proposal_prompt(agent, audience_section)

    def _build_revision_prompt(self, agent: Agent, original: str, critiques: list[Critique]) -> str:
        """Build the revision prompt including critiques."""
        audience_section = self._prepare_audience_context(emit_event=False)
        return self.prompt_builder.build_revision_prompt(
            agent, original, critiques, audience_section
        )
