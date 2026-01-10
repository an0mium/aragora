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
from dataclasses import dataclass
from typing import Optional

from aragora.audience.suggestions import cluster_suggestions, format_for_prompt
from aragora.core import Agent, Critique, DebateResult, DisagreementReport, Environment, Message, Vote
from aragora.debate.convergence import ConvergenceDetector
from aragora.debate.disagreement import DisagreementReporter
from aragora.debate.event_bridge import EventEmitterBridge
from aragora.debate.immune_system import TransparentImmuneSystem, get_immune_system
from aragora.debate.chaos_theater import ChaosDirector, get_chaos_director, DramaLevel
from aragora.debate.audience_manager import AudienceManager
from aragora.debate.autonomic_executor import AutonomicExecutor
from aragora.debate.arena_phases import init_phases
from aragora.debate.complexity_governor import (
    classify_task_complexity,
    get_complexity_governor,
)
from aragora.debate.optional_imports import OptionalImports
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.debate.roles import (
    RoleAssignment,
    RoleRotationConfig,
    RoleRotator,
)
from aragora.debate.role_matcher import RoleMatcher, RoleMatchingConfig
from aragora.debate.topology import TopologySelector
from aragora.debate.judge_selector import JudgeSelector, JudgeScoringMixin
from aragora.debate.sanitization import OutputSanitizer
from aragora.spectate.stream import SpectatorStream

from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


@dataclass
class ArenaConfig:
    """Configuration for Arena debate orchestration.

    Groups optional dependencies and settings that can be passed to Arena.
    This allows for cleaner initialization and easier testing.

    Initialization Flow
    -------------------
    Arena initialization follows a layered architecture pattern:

    1. **Core Configuration** (_init_core):
       - Sets up environment, agents, protocol, spectator
       - Initializes circuit breaker for fault tolerance
       - Configures loop scoping for multi-debate sessions

    2. **Tracking Subsystems** (_init_trackers):
       - Position and belief tracking (PositionTracker, PositionLedger)
       - ELO rating system for agent ranking
       - Persona manager for agent specialization
       - Flip detector for position reversals
       - Relationship tracker for agent interactions
       - Moment detector for significant events

    3. **User Participation** (_init_user_participation):
       - Sets up event queue for user votes/suggestions
       - Subscribes to event emitter for real-time participation

    4. **Roles and Stances** (_init_roles_and_stances):
       - Initializes cognitive role rotation (Heavy3-inspired)
       - Sets up initial agent stances

    5. **Convergence Detection** (_init_convergence):
       - Configures semantic similarity backend
       - Sets up convergence thresholds

    6. **Phase Classes** (_init_phases):
       - ContextInitializer, ProposalPhase, DebateRoundsPhase
       - ConsensusPhase, AnalyticsPhase, FeedbackPhase, VotingPhase

    Dependency Injection
    --------------------
    Most subsystems are optional and can be injected for testing:

    - Memory systems: critique_store, continuum_memory, debate_embeddings
    - Tracking: elo_system, position_ledger, relationship_tracker
    - Events: event_emitter, spectator
    - Recording: recorder, evidence_collector

    Example
    -------
    Basic usage with minimal configuration::

        config = ArenaConfig(loop_id="debate-123")
        arena = Arena.from_config(env, agents, protocol, config)

    Full production setup with all subsystems::

        config = ArenaConfig(
            loop_id="debate-123",
            strict_loop_scoping=True,
            memory=critique_store,
            continuum_memory=continuum,
            elo_system=elo,
            event_emitter=emitter,
            spectator=stream,
        )
        arena = Arena.from_config(env, agents, protocol, config)
    """

    # Identification
    loop_id: str = ""

    # Behavior flags
    strict_loop_scoping: bool = False

    # Core subsystems (typically injected)
    memory: Optional[object] = None  # CritiqueStore
    event_hooks: Optional[dict] = None
    event_emitter: Optional[object] = None
    spectator: Optional[SpectatorStream] = None
    debate_embeddings: Optional[object] = None  # DebateEmbeddingsDatabase
    insight_store: Optional[object] = None  # InsightStore
    recorder: Optional[object] = None  # ReplayRecorder
    circuit_breaker: Optional[CircuitBreaker] = None
    evidence_collector: Optional[object] = None

    # Agent configuration
    agent_weights: Optional[dict] = None

    # Tracking subsystems
    position_tracker: Optional[object] = None
    position_ledger: Optional[object] = None
    enable_position_ledger: bool = False  # Auto-create PositionLedger if not provided
    elo_system: Optional[object] = None
    persona_manager: Optional[object] = None
    dissent_retriever: Optional[object] = None
    consensus_memory: Optional[object] = None  # ConsensusMemory for historical outcomes
    flip_detector: Optional[object] = None
    calibration_tracker: Optional[object] = None
    continuum_memory: Optional[object] = None
    relationship_tracker: Optional[object] = None
    moment_detector: Optional[object] = None

    # Genesis evolution
    population_manager: Optional[object] = None  # PopulationManager for genome evolution
    auto_evolve: bool = False  # Trigger evolution after high-quality debates
    breeding_threshold: float = 0.8  # Min confidence to trigger evolution

    # Fork/continuation support
    initial_messages: Optional[list] = None
    trending_topic: Optional[object] = None
    pulse_manager: Optional[object] = None  # PulseManager for auto-fetching trending topics
    auto_fetch_trending: bool = False  # Auto-fetch trending topics if none provided

    # Human-in-the-loop breakpoints
    breakpoint_manager: Optional[object] = None  # BreakpointManager

    # Performance telemetry
    performance_monitor: Optional[object] = None  # AgentPerformanceMonitor
    enable_performance_monitor: bool = False
    enable_telemetry: bool = False  # Enable Prometheus/Blackbox telemetry emission

    # Agent selection (performance-based team formation)
    agent_selector: Optional[object] = None  # AgentSelector for performance-based selection
    use_performance_selection: bool = False  # Enable ELO/calibration-based agent selection

    # Airlock resilience layer
    use_airlock: bool = False  # Wrap agents with AirlockProxy for timeout/fallback
    airlock_config: Optional[object] = None  # AirlockConfig for customization

    # Prompt evolution for self-improvement
    prompt_evolver: Optional[object] = None  # PromptEvolver for extracting winning patterns
    enable_prompt_evolution: bool = False  # Auto-create PromptEvolver if True


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
        agent_weights: dict[str, float] | None = None,  # Optional reliability weights from capability probing
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
        performance_monitor=None,  # Optional AgentPerformanceMonitor for telemetry
        enable_performance_monitor: bool = False,  # Auto-create PerformanceMonitor if True
        enable_telemetry: bool = False,  # Enable Prometheus/Blackbox telemetry emission
        use_airlock: bool = False,  # Wrap agents with AirlockProxy for timeout protection
        airlock_config=None,  # Optional AirlockConfig for customization
        agent_selector=None,  # Optional AgentSelector for performance-based team selection
        use_performance_selection: bool = False,  # Enable ELO/calibration-based agent selection
        prompt_evolver=None,  # Optional PromptEvolver for extracting winning patterns
        enable_prompt_evolution: bool = False,  # Auto-create PromptEvolver if True
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
            performance_monitor=performance_monitor,
            enable_performance_monitor=enable_performance_monitor,
            enable_telemetry=enable_telemetry,
            use_airlock=use_airlock,
            airlock_config=airlock_config,
            agent_selector=agent_selector,
            use_performance_selection=use_performance_selection,
            prompt_evolver=prompt_evolver,
            enable_prompt_evolution=enable_prompt_evolution,
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
        )

        # Initialize user participation and roles
        self._init_user_participation()
        self._init_roles_and_stances()

        # Initialize convergence detection and caches
        self._init_convergence()
        self._init_caches()

        # Initialize phase classes for orchestrator decomposition
        self._init_phases()

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
        performance_monitor,
        enable_performance_monitor: bool,
        enable_telemetry: bool,
        use_airlock: bool,
        airlock_config,
        agent_selector,
        use_performance_selection: bool,
        prompt_evolver,
        enable_prompt_evolution: bool,
    ) -> None:
        """Initialize core Arena configuration."""
        self.env = environment
        self.agents = agents
        self.protocol = protocol or DebateProtocol()

        # Wrap agents with airlock protection if enabled
        if use_airlock:
            from aragora.agents.airlock import wrap_agents, AirlockConfig
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
        elif enable_prompt_evolution:
            from aragora.evolution.evolver import PromptEvolver
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

        # Connect immune system to event bridge for WebSocket broadcasting
        self.immune_system.set_broadcast_callback(self._broadcast_health_event)

    def _broadcast_health_event(self, event: dict) -> None:
        """Broadcast health events to WebSocket clients via event bridge."""
        try:
            self.event_bridge.notify(
                event_type="health_event",
                **event.get("data", event),
            )
        except Exception as e:
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
        except Exception as e:
            logger.warning("PositionLedger auto-init failed: %s", e)

    def _auto_init_calibration_tracker(self) -> None:
        """Auto-initialize CalibrationTracker when enable_calibration is True."""
        try:
            from aragora.agents.calibration import CalibrationTracker
            self.calibration_tracker = CalibrationTracker()
            logger.debug("Auto-initialized CalibrationTracker for prediction calibration")
        except ImportError:
            logger.warning("CalibrationTracker not available - calibration disabled")
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            logger.debug("MomentDetector auto-init failed: %s", e)

    def _auto_init_breakpoint_manager(self) -> None:
        """Auto-initialize BreakpointManager when enable_breakpoints is True."""
        try:
            from aragora.debate.breakpoints import BreakpointManager, BreakpointConfig

            config = self.protocol.breakpoint_config or BreakpointConfig()
            self.breakpoint_manager = BreakpointManager(config=config)
            logger.debug("Auto-initialized BreakpointManager for human-in-the-loop breakpoints")
        except ImportError:
            logger.warning("BreakpointManager not available - breakpoints disabled")
        except Exception as e:
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

    @property
    def user_votes(self) -> deque[dict]:
        """Get user votes from AudienceManager (backward compatibility)."""
        return self.audience_manager._votes

    @property
    def user_suggestions(self) -> deque[dict]:
        """Get user suggestions from AudienceManager (backward compatibility)."""
        return self.audience_manager._suggestions

    def _init_roles_and_stances(self) -> None:
        """Initialize cognitive role rotation and agent stances."""
        # Cognitive role rotation (Heavy3-inspired)
        self.role_rotator: Optional[RoleRotator] = None
        self.role_matcher: Optional[RoleMatcher] = None
        self.current_role_assignments: dict[str, RoleAssignment] = {}

        # Role matching takes priority over simple rotation
        if self.protocol.role_matching:
            config = self.protocol.role_matching_config or RoleMatchingConfig()
            self.role_matcher = RoleMatcher(
                calibration_tracker=self.calibration_tracker if hasattr(self, 'calibration_tracker') else None,
                persona_manager=self.persona_manager if hasattr(self, 'persona_manager') else None,
                config=config,
            )
            logger.info("role_matcher_enabled strategy=%s", config.strategy)
        elif self.protocol.role_rotation:
            config = self.protocol.role_rotation_config or RoleRotationConfig()
            self.role_rotator = RoleRotator(config)

        # Assign roles if not already set
        self._assign_roles()

        # Assign initial stances for asymmetric debate
        self._assign_stances(round_num=0)

        # Apply agreement intensity guidance to all agents
        self._apply_agreement_intensity()

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

        # Cached similarity backend for vote grouping (avoids recreating per call)
        self._similarity_backend = None

        # Cache for debate domain (computed once per debate)
        self._debate_domain_cache: Optional[str] = None

    def _init_phases(self) -> None:
        """Initialize phase classes for orchestrator decomposition."""
        init_phases(self)

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

        Uses the debate task and domain to query for related past learnings.
        Enhanced with tier-aware retrieval and confidence markers.
        """
        if self._continuum_context_cache:
            return self._continuum_context_cache

        if not self.continuum_memory:
            return ""

        try:
            domain = self._extract_debate_domain()
            query = f"{domain}: {self.env.task[:200]}"

            # Retrieve memories, prioritizing fast/medium tiers (skip glacial for speed)
            memories = self.continuum_memory.retrieve(
                query=query,
                limit=5,
                min_importance=0.3,  # Only important memories
            )

            if not memories:
                return ""

            # Track retrieved memory IDs for outcome updates after debate
            self._continuum_retrieved_ids = [
                getattr(mem, 'id', None) for mem in memories if getattr(mem, 'id', None)
            ]

            # Format memories with confidence markers based on consolidation
            context_parts = ["[Previous learnings relevant to this debate:]"]
            for mem in memories[:3]:  # Top 3 most relevant
                content = mem.content[:200] if hasattr(mem, 'content') else str(mem)[:200]
                tier = mem.tier.value if hasattr(mem, 'tier') else "unknown"
                # Consolidation score indicates reliability
                consolidation = getattr(mem, 'consolidation_score', 0.5)
                confidence = "high" if consolidation > 0.7 else "medium" if consolidation > 0.4 else "low"
                context_parts.append(f"- [{tier}|{confidence}] {content}")

            self._continuum_context_cache = "\n".join(context_parts)
            logger.info(f"  [continuum] Retrieved {len(memories)} relevant memories for domain '{domain}'")
            return self._continuum_context_cache
        except (AttributeError, TypeError, ValueError) as e:
            # Expected errors from memory system
            logger.warning(f"  [continuum] Memory retrieval error: {e}")
            return ""
        except Exception as e:
            # Unexpected error - log with more detail but don't crash debate
            logger.warning(f"  [continuum] Unexpected memory error (type={type(e).__name__}): {e}")
            return ""

    def _store_debate_outcome_as_memory(self, result: "DebateResult") -> None:
        """Store debate outcome in ContinuumMemory for future retrieval."""
        # Extract belief cruxes from result if set by AnalyticsPhase
        belief_cruxes = getattr(result, 'belief_cruxes', None)
        if belief_cruxes:
            belief_cruxes = [str(c) for c in belief_cruxes[:10]]
        self.memory_manager.store_debate_outcome(result, self.env.task, belief_cruxes=belief_cruxes)

    def _store_evidence_in_memory(self, evidence_snippets: list, task: str) -> None:
        """Store collected evidence snippets in ContinuumMemory for future retrieval."""
        self.memory_manager.store_evidence(evidence_snippets, task)

    def _update_continuum_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome."""
        # Sync tracked IDs to memory manager
        self.memory_manager.track_retrieved_ids(self._continuum_retrieved_ids)
        self.memory_manager.update_memory_outcomes(result)
        # Clear local tracking
        self._continuum_retrieved_ids = []

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
        Result is cached since the task doesn't change during a debate.
        """
        # Return cached domain if available
        if self._debate_domain_cache is not None:
            return self._debate_domain_cache

        task_lower = self.env.task.lower()

        # Domain detection heuristics
        if any(w in task_lower for w in ["security", "hack", "vulnerability", "auth", "encrypt"]):
            domain = "security"
        elif any(w in task_lower for w in ["performance", "speed", "optimize", "cache", "latency"]):
            domain = "performance"
        elif any(w in task_lower for w in ["test", "testing", "coverage", "regression"]):
            domain = "testing"
        elif any(w in task_lower for w in ["design", "architecture", "pattern", "structure"]):
            domain = "architecture"
        elif any(w in task_lower for w in ["bug", "error", "fix", "crash", "exception"]):
            domain = "debugging"
        elif any(w in task_lower for w in ["api", "endpoint", "rest", "graphql"]):
            domain = "api"
        elif any(w in task_lower for w in ["database", "sql", "query", "schema"]):
            domain = "database"
        elif any(w in task_lower for w in ["ui", "frontend", "react", "css", "layout"]):
            domain = "frontend"
        else:
            domain = "general"

        # Cache and return
        self._debate_domain_cache = domain
        return domain

    def _select_debate_team(self, requested_agents: list[Agent]) -> list[Agent]:
        """Select debate team using performance metrics if enabled.

        When use_performance_selection is True and an agent_selector is configured,
        agents are scored based on ELO ratings, calibration scores, and domain
        expertise, then sorted by performance. This allows high-performing agents
        to be prioritized for the debate.

        Args:
            requested_agents: Original list of agents requested for the debate

        Returns:
            Sorted list of agents, prioritized by performance if enabled,
            otherwise the original list unchanged.
        """
        if not self.use_performance_selection:
            return requested_agents

        if not self.agent_selector:
            logger.debug("performance_selection enabled but no agent_selector configured")
            return requested_agents

        # Get domain for task-specific scoring
        domain = self._extract_debate_domain()

        # Filter out unavailable agents via circuit breaker
        available_names = {a.name for a in requested_agents}
        if self.circuit_breaker:
            try:
                available_names = set(
                    self.circuit_breaker.filter_available_agents(
                        [a.name for a in requested_agents]
                    )
                )
            except Exception as e:
                logger.debug(f"circuit_breaker filter error: {e}")

        # Score agents using ELO and calibration
        scored: list[tuple[Agent, float]] = []
        for agent in requested_agents:
            if agent.name not in available_names:
                logger.info(f"agent_filtered_by_circuit_breaker agent={agent.name}")
                continue

            score = 1.0  # Base score

            # ELO contribution (if available)
            if self.elo_system:
                try:
                    elo = self.elo_system.get_rating(agent.name)
                    # Normalize: 1000 is average, each 100 points = 0.1 bonus
                    score += (elo - 1000) / 1000 * 0.3
                except Exception:
                    pass

            # Calibration contribution (well-calibrated agents get a bonus)
            if self.calibration_tracker:
                try:
                    brier = self.calibration_tracker.get_brier_score(agent.name)
                    # Lower Brier = better calibration = higher score
                    # Brier ranges 0-1, so (1 - brier) gives 0-1 bonus
                    score += (1 - brier) * 0.2
                except Exception:
                    pass

            scored.append((agent, score))

        if not scored:
            logger.warning("No agents available after performance filtering")
            return requested_agents

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [agent for agent, _ in scored]
        logger.info(
            f"performance_selection domain={domain} "
            f"selected={[a.name for a in selected]} "
            f"scores={[f'{s:.2f}' for _, s in scored]}"
        )

        return selected

    def _get_calibration_weight(self, agent_name: str) -> float:
        """Get agent weight based on calibration score (0.5-1.5 range).

        Delegates to JudgeScoringMixin. For new code, use:
            JudgeScoringMixin(elo_system).get_calibration_weight(agent_name)
        """
        scorer = JudgeScoringMixin(self.elo_system)
        return scorer.get_calibration_weight(agent_name)

    def _compute_composite_judge_score(self, agent_name: str) -> float:
        """Compute composite score for judge selection (ELO + calibration).

        Delegates to JudgeScoringMixin. For new code, use:
            JudgeScoringMixin(elo_system).compute_composite_score(agent_name)
        """
        scorer = JudgeScoringMixin(self.elo_system)
        return scorer.compute_composite_score(agent_name)

    def _select_critics_for_proposal(self, proposal_agent: str, all_critics: list[Agent]) -> list[Agent]:
        """Select which critics should critique the given proposal based on topology.

        Delegates to TopologySelector for cleaner topology strategy implementation.
        """
        selector = TopologySelector.from_protocol(self.protocol, self._require_agents())
        return selector.select_critics(proposal_agent, all_critics)

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
        """Delegate to event bridge for spectator/websocket emission."""
        self.event_bridge.notify(event_type, **kwargs)

    def _emit_moment_event(self, moment) -> None:
        """Delegate to event bridge for moment emission."""
        self.event_bridge.emit_moment(moment)

    def _record_grounded_position(
        self, agent_name: str, content: str, debate_id: str, round_num: int,
        confidence: float = 0.7, domain: Optional[str] = None,
    ):
        """Record a position to the grounded persona ledger."""
        if not self.position_ledger:
            return
        try:
            self.position_ledger.record_position(
                agent_name=agent_name, claim=content[:1000], confidence=confidence,
                debate_id=debate_id, round_num=round_num, domain=domain,
            )
        except (AttributeError, TypeError, ValueError) as e:
            # Expected parameter or state errors
            logger.warning(f"Position ledger error: {e}")
        except Exception as e:
            # Unexpected error - log type for debugging
            logger.warning(f"Position ledger error (type={type(e).__name__}): {e}")

    def _update_agent_relationships(self, debate_id: str, participants: list[str], winner: Optional[str], votes: list):
        """Update agent relationships after debate completion.

        Uses batch update for O(1) database connections instead of O(n²) for n participants.
        """
        if not self.elo_system:
            return
        try:
            vote_choices = {v.agent: v.choice for v in votes if hasattr(v, 'agent') and hasattr(v, 'choice')}
            # Build batch of relationship updates
            updates = []
            for i, agent_a in enumerate(participants):
                for agent_b in participants[i + 1:]:
                    agreed = agent_a in vote_choices and agent_b in vote_choices and vote_choices[agent_a] == vote_choices[agent_b]
                    a_win = 1 if winner == agent_a else 0
                    b_win = 1 if winner == agent_b else 0
                    updates.append({
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "debate_increment": 1,
                        "agreement_increment": 1 if agreed else 0,
                        "a_win": a_win,
                        "b_win": b_win,
                    })
            # Single transaction for all updates
            self.elo_system.update_relationships_batch(updates)
        except (AttributeError, TypeError, KeyError) as e:
            # Expected data access errors
            logger.warning(f"Relationship update error: {e}")
        except Exception as e:
            # Unexpected error - log type for debugging
            logger.warning(f"Relationship update error (type={type(e).__name__}): {e}")

    def _generate_disagreement_report(
        self,
        votes: list[Vote],
        critiques: list[Critique],
        winner: Optional[str] = None,
    ) -> DisagreementReport:
        """
        Generate a DisagreementReport from debate votes and critiques.

        Delegates to DisagreementReporter for the actual analysis.
        """
        reporter = DisagreementReporter()
        return reporter.generate_report(votes, critiques, winner)

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

    def _format_conclusion(self, result: "DebateResult") -> str:
        """Format a clear, readable debate conclusion with full context."""
        lines = []
        lines.append("=" * 60)
        lines.append("DEBATE CONCLUSION")
        lines.append("=" * 60)

        # Verdict section
        lines.append("\n## VERDICT")
        if result.consensus_reached:
            lines.append(f"Consensus: YES ({result.confidence:.0%} agreement)")
            if hasattr(result, 'consensus_strength') and result.consensus_strength:
                lines.append(f"Strength: {result.consensus_strength.upper()}")
        else:
            lines.append(f"Consensus: NO (only {result.confidence:.0%} agreement)")

        # Winner (if determined)
        if hasattr(result, 'winner') and result.winner:
            lines.append(f"Winner: {result.winner}")

        # Final answer section
        lines.append("\n## FINAL ANSWER")
        if result.final_answer:
            # Truncate if very long, but show substantial content
            answer_display = result.final_answer[:1000] + "..." if len(result.final_answer) > 1000 else result.final_answer
            lines.append(answer_display)
        else:
            lines.append("No final answer determined.")

        # Vote breakdown (if available)
        if hasattr(result, 'votes') and result.votes:
            lines.append("\n## VOTE BREAKDOWN")
            vote_counts = {}
            for vote in result.votes:
                voter = getattr(vote, 'voter', 'unknown')
                choice = getattr(vote, 'choice', 'abstain')
                vote_counts[voter] = choice
            for voter, choice in vote_counts.items():
                lines.append(f"  - {voter}: {choice}")

        # Dissenting views (if any)
        if hasattr(result, 'dissenting_views') and result.dissenting_views:
            lines.append("\n## DISSENTING VIEWS")
            for i, view in enumerate(result.dissenting_views[:3]):
                view_display = view[:300] + "..." if len(view) > 300 else view
                lines.append(f"  {i+1}. {view_display}")

        # Debate cruxes (key disagreement points)
        if hasattr(result, 'belief_cruxes') and result.belief_cruxes:
            lines.append("\n## KEY CRUXES")
            for crux in result.belief_cruxes[:3]:
                claim = crux.get('claim', 'unknown')[:80]
                uncertainty = crux.get('uncertainty', 0)
                lines.append(f"  - {claim}... (uncertainty: {uncertainty:.2f})")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def _assign_roles(self):
        """Assign roles to agents based on protocol with safety bounds."""
        # If agents already have roles, respect them
        if all(a.role for a in self.agents):
            return

        n_agents = len(self.agents)

        # Safety: Ensure at least 1 critic and 1 synthesizer when we have 3+ agents
        # This prevents all-proposer scenarios that break debate dynamics
        max_proposers = max(1, n_agents - 2) if n_agents >= 3 else 1
        proposers_needed = min(self.protocol.proposer_count, max_proposers)

        for i, agent in enumerate(self.agents):
            if i < proposers_needed:
                agent.role = "proposer"
            elif i == n_agents - 1:
                agent.role = "synthesizer"
            else:
                agent.role = "critic"

        # Log role assignment for debugging
        roles = {a.name: a.role for a in self.agents}
        logger.debug(f"Role assignment: {roles}")

    def _apply_agreement_intensity(self):
        """Apply agreement intensity guidance to all agents' system prompts.

        This modifies each agent's system_prompt to include guidance on how
        much to agree vs disagree with other agents, based on the protocol's
        agreement_intensity setting.
        """
        guidance = self._get_agreement_intensity_guidance()

        for agent in self.agents:
            if agent.system_prompt:
                agent.system_prompt = f"{agent.system_prompt}\n\n{guidance}"
            else:
                agent.system_prompt = guidance

    def _assign_stances(self, round_num: int = 0):
        """Assign debate stances to agents for asymmetric debate.

        Stances: "affirmative" (defend), "negative" (challenge), "neutral" (evaluate)
        If rotate_stances is True, stances rotate each round.
        """
        if not self.protocol.asymmetric_stances:
            return

        stances = ["affirmative", "negative", "neutral"]
        n_agents = len(self.agents)

        for i, agent in enumerate(self.agents):
            # Rotate stance based on round number if enabled
            if self.protocol.rotate_stances:
                stance_idx = (i + round_num) % len(stances)
            else:
                stance_idx = i % len(stances)

            agent.stance = stances[stance_idx]

    def _get_stance_guidance(self, agent) -> str:
        """Generate prompt guidance based on agent's debate stance.

        Delegates to PromptBuilder for the actual implementation.
        """
        return self.prompt_builder.get_stance_guidance(agent)

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
                    timeout=self.protocol.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"debate_timeout timeout_seconds={self.protocol.timeout_seconds}")
                # Return partial result with timeout indicator
                return DebateResult(
                    task=self.env.task,
                    messages=getattr(self, '_partial_messages', []),
                    critiques=getattr(self, '_partial_critiques', []),
                    votes=[],
                    dissenting_views=[],
                    rounds_used=getattr(self, '_partial_rounds', 0),
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

        # Create shared context for all phases
        ctx = DebateContext(
            env=self.env,
            agents=self.agents,
            start_time=time.time(),
            debate_id=debate_id,
            correlation_id=correlation_id,
            domain=self._extract_debate_domain(),
        )

        # Classify task complexity and configure adaptive timeouts
        task_complexity = classify_task_complexity(self.env.task)
        governor = get_complexity_governor()
        governor.set_task_complexity(task_complexity)

        # Apply performance-based agent selection if enabled
        if self.use_performance_selection:
            self.agents = self._select_debate_team(self.agents)
            ctx.agents = self.agents  # Update context with selected agents

        logger.info(
            f"debate_start id={debate_id[:8]} complexity={task_complexity.value} "
            f"agents={[a.name for a in self.agents]}"
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

        try:
            # Phase 0: Context Initialization
            await self.context_initializer.initialize(ctx)

            # Phase 1: Initial Proposals
            await self.proposal_phase.execute(ctx)

            # Phase 2: Debate Rounds (critique/revision loop)
            await self.debate_rounds_phase.execute(ctx)

            # Phase 3: Consensus Resolution
            await self.consensus_phase.execute(ctx)

            # Phases 4-6: Analytics
            await self.analytics_phase.execute(ctx)

            # Phase 7: Feedback Loops
            await self.feedback_phase.execute(ctx)

        except asyncio.TimeoutError:
            # Timeout recovery - use partial results from context
            ctx.result.messages = ctx.partial_messages
            ctx.result.critiques = ctx.partial_critiques
            ctx.result.rounds_used = ctx.partial_rounds
            logger.warning("Debate timed out, returning partial results")

        return ctx.result

    # NOTE: Legacy _run_inner code (1,300+ lines) removed after successful phase integration.
    # The debate execution is now handled by phase classes:
    # - ContextInitializer (Phase 0)
    # - ProposalPhase (Phase 1)
    # - DebateRoundsPhase (Phase 2)
    # - ConsensusPhase (Phase 3)
    # - AnalyticsPhase (Phases 4-6)
    # - FeedbackPhase (Phase 7)

    async def _index_debate_async(self, artifact: dict) -> None:
        """Index debate asynchronously to avoid blocking."""
        try:
            if self.debate_embeddings:
                await self.debate_embeddings.index_debate(artifact)
        except Exception as e:
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
        """
        Have a judge evaluate if the debate is conclusive.

        Returns:
            Tuple of (should_continue: bool, reason: str)
        """
        if not self.protocol.judge_termination:
            return True, ""

        if round_num < self.protocol.min_rounds_before_judge_check:
            return True, ""

        # Select a judge (use existing method)
        judge = await self._select_judge(proposals, context)

        prompt = f"""You are evaluating whether this multi-agent debate has reached a conclusive state.

Task: {self.env.task[:300]}

After {round_num} rounds of debate, the proposals are:
{chr(10).join(f"- {agent}: {prop[:200]}..." for agent, prop in proposals.items())}

Evaluate:
1. Have the key issues been thoroughly discussed?
2. Are there major unresolved disagreements that more debate could resolve?
3. Would additional rounds likely produce meaningful improvements?

Respond with:
CONCLUSIVE: <yes/no>
REASON: <brief explanation>"""

        try:
            response = await self.autonomic.generate(judge, prompt, context[-5:])
            lines = response.strip().split('\n')

            conclusive = False
            reason = ""

            for line in lines:
                if line.upper().startswith("CONCLUSIVE:"):
                    val = line.split(":", 1)[1].strip().lower()
                    conclusive = val in ("yes", "true", "1")
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            if conclusive:
                logger.info(f"judge_termination judge={judge.name} reason={reason[:100]}")
                # Emit event
                if "on_judge_termination" in self.hooks:
                    self.hooks["on_judge_termination"](judge.name, reason)
                return False, reason

        except Exception as e:
            logger.warning(f"Judge termination check failed: {e}")

        return True, ""

    async def _check_early_stopping(
        self, round_num: int, proposals: dict[str, str], context: list[Message]
    ) -> bool:
        """Check if agents want to stop debate early.

        Returns True if debate should continue, False if it should stop.
        """
        if not self.protocol.early_stopping:
            return True  # Continue

        if round_num < self.protocol.min_rounds_before_early_stop:
            return True  # Continue - haven't met minimum rounds

        # Ask each agent if they think more debate would help
        prompt = f"""After {round_num} round(s) of debate on this task:
Task: {self.env.task[:200]}

Current proposals have been critiqued and revised. Do you think additional debate
rounds would significantly improve the answer quality?

Respond with only: CONTINUE or STOP
- CONTINUE: More debate rounds would help refine the answer
- STOP: The proposals are mature enough, further debate is unlikely to help"""

        stop_votes = 0
        total_votes = 0

        tasks = [self.autonomic.generate(agent, prompt, context[-5:]) for agent in self.agents]
        try:
            # Use wait_for for Python 3.10 compatibility (asyncio.timeout is 3.11+)
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.protocol.round_timeout_seconds
            )
        except asyncio.TimeoutError:
            # Timeout during early stopping check - continue debate (safe default)
            logger.warning(f"Early stopping check timed out after {self.protocol.round_timeout_seconds}s")
            return True

        for agent, result in zip(self.agents, results):
            if isinstance(result, BaseException):
                continue
            total_votes += 1
            response = str(result).strip().upper()
            if "STOP" in response and "CONTINUE" not in response:
                stop_votes += 1

        if total_votes == 0:
            return True  # Continue if voting failed

        stop_ratio = stop_votes / total_votes
        should_stop = stop_ratio >= self.protocol.early_stop_threshold

        if should_stop:
            logger.info(f"early_stopping votes={stop_votes}/{total_votes}")
            # Emit early stop event
            if "on_early_stop" in self.hooks:
                self.hooks["on_early_stop"](round_num, stop_votes, total_votes)

        return not should_stop  # Return True to continue, False to stop

    async def _select_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Select judge based on protocol.judge_selection setting.

        Delegates to JudgeSelector. For new code, use:
            selector = JudgeSelector.from_protocol(protocol, agents, elo_system, ...)
            judge = await selector.select_judge(proposals, context)
        """
        async def generate_wrapper(agent, prompt, ctx):
            return await agent.generate(prompt, ctx)

        selector = JudgeSelector(
            agents=self._require_agents(),
            elo_system=self.elo_system,
            judge_selection=self.protocol.judge_selection,
            generate_fn=generate_wrapper,
            build_vote_prompt_fn=lambda candidates, props: self.prompt_builder.build_judge_vote_prompt(candidates, props),
            sanitize_fn=OutputSanitizer.sanitize_agent_output,
            consensus_memory=self.consensus_memory,
        )
        return await selector.select_judge(proposals, context)

    async def _vote_for_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Have agents vote on who should be the judge.

        Delegates to JudgeSelector with voted strategy.
        """
        async def generate_wrapper(agent, prompt, ctx):
            return await agent.generate(prompt, ctx)

        selector = JudgeSelector(
            agents=self._require_agents(),
            elo_system=self.elo_system,
            judge_selection="voted",
            generate_fn=generate_wrapper,
            build_vote_prompt_fn=lambda candidates, props: self.prompt_builder.build_judge_vote_prompt(candidates, props),
            sanitize_fn=OutputSanitizer.sanitize_agent_output,
            consensus_memory=self.consensus_memory,
        )
        return await selector.select_judge(proposals, context)

    def _build_judge_vote_prompt(self, candidates: list[Agent], proposals: dict[str, str]) -> str:
        """Build prompt for voting on who should judge."""
        return self.prompt_builder.build_judge_vote_prompt(candidates, proposals)

    def _get_agreement_intensity_guidance(self) -> str:
        """Generate prompt guidance based on agreement intensity setting.

        Agreement intensity (0-10) affects how agents approach disagreements:
        - Low (0-3): Adversarial - strongly challenge others' positions
        - Medium (4-6): Balanced - judge arguments on merit
        - High (7-10): Collaborative - seek common ground and synthesis
        """
        intensity = self.protocol.agreement_intensity

        if intensity is None:
            return ""  # No agreement intensity guidance when not set

        if intensity <= 1:
            return """IMPORTANT: You strongly disagree with other agents. Challenge every assumption,
find flaws in every argument, and maintain your original position unless presented
with irrefutable evidence. Be adversarial but constructive."""
        elif intensity <= 3:
            return """IMPORTANT: Approach others' arguments with healthy skepticism. Be critical of
proposals and require strong evidence before changing your position. Point out
weaknesses even if you partially agree."""
        elif intensity <= 6:
            return """Evaluate arguments on their merits. Agree when others make valid points,
disagree when you see genuine flaws. Let the quality of reasoning guide your response."""
        elif intensity <= 8:
            return """Look for common ground with other agents. Acknowledge valid points in others'
arguments and try to build on them. Seek synthesis where possible while maintaining
your own reasoned perspective."""
        else:  # 9-10
            return """Actively seek to incorporate other agents' perspectives. Find value in all
proposals and work toward collaborative synthesis. Prioritize finding agreement
and building on others' ideas."""

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
                issue_preview = p.issue_text[:100] + "..." if len(p.issue_text) > 100 else p.issue_text
                fix_preview = p.suggestion_text[:80] + "..." if len(p.suggestion_text) > 80 else p.suggestion_text
                lines.append(f"- **{p.issue_type}**: {issue_preview}")
                if fix_preview:
                    lines.append(f"  Fix: {fix_preview} ({p.success_count} successes)")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Pattern retrieval error: {e}")
            return ""

    def _update_role_assignments(self, round_num: int) -> None:
        """Update cognitive role assignments for the current round."""
        agent_names = [a.name for a in self.agents]

        # Use role matcher if available (calibration-based)
        if self.role_matcher:
            debate_domain = getattr(self, 'current_domain', None)
            result = self.role_matcher.match_roles(
                agent_names=agent_names,
                round_num=round_num,
                debate_domain=debate_domain,
            )
            self.current_role_assignments = result.assignments

            if result.assignments:
                roles_str = ", ".join(
                    f"{name}: {assign.role.value}"
                    for name, assign in result.assignments.items()
                )
                logger.debug(
                    f"role_assignments round={round_num} strategy={result.strategy_used} "
                    f"roles={roles_str}"
                )
                if result.developmental_assignments:
                    logger.debug(
                        f"developmental_assignments agents={result.developmental_assignments}"
                    )
            return

        # Fallback to simple rotation
        if not self.role_rotator:
            return

        self.current_role_assignments = self.role_rotator.get_assignments(
            agent_names, round_num, self.protocol.rounds
        )

        if self.current_role_assignments:
            roles_str = ", ".join(
                f"{name}: {assign.role.value}"
                for name, assign in self.current_role_assignments.items()
            )
            logger.debug(f"role_assignments round={round_num} roles={roles_str}")

    def _get_role_context(self, agent: Agent) -> str:
        """Get cognitive role context for an agent in the current round."""
        if not self.role_rotator or agent.name not in self.current_role_assignments:
            return ""

        assignment = self.current_role_assignments[agent.name]
        return self.role_rotator.format_role_context(assignment)

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

        except Exception as e:
            logger.warning(f"Flip context error for {agent.name}: {e}")
            return ""

    async def _check_breakpoint(
        self,
        round_num: int,
        proposals: dict[str, str],
        context: list,
        confidence: float = 0.0,
    ) -> Optional[str]:
        """
        Check for breakpoint triggers and handle human intervention if needed.

        Args:
            round_num: Current debate round
            proposals: Current proposals from agents
            context: Conversation context
            confidence: Current consensus confidence (0.0-1.0)

        Returns:
            Optional guidance string from human if breakpoint triggered,
            None otherwise
        """
        if not self.breakpoint_manager or not self.protocol.enable_breakpoints:
            return None

        try:
            from aragora.debate.breakpoints import BreakpointTrigger, DebateSnapshot

            # Create debate snapshot for breakpoint evaluation
            snapshot = DebateSnapshot(
                debate_id=self.env.debate_id if hasattr(self.env, 'debate_id') else "",
                task=self.env.task,
                current_round=round_num,
                total_rounds=self.protocol.rounds,
                latest_messages=[],  # Would need message history
                active_proposals=list(proposals.values()) if isinstance(proposals, dict) else proposals,
                open_critiques=[],
                current_consensus=None,
                confidence=confidence,
                agent_positions={a.name: "" for a in self.agents},
                unresolved_issues=[],
                key_disagreements=[],
            )

            # Check for triggers
            breakpoint = self.breakpoint_manager.check_triggers(
                snapshot=snapshot,
                confidence=confidence,
                round_num=round_num,
                total_rounds=self.protocol.rounds,
            )

            if breakpoint:
                # Emit breakpoint event
                self.event_bridge.notify(
                    event_type="breakpoint",
                    breakpoint_id=breakpoint.breakpoint_id,
                    trigger=breakpoint.trigger.value,
                    round=round_num,
                    message=breakpoint.message,
                )

                # Wait for human guidance
                guidance = await self.breakpoint_manager.handle_breakpoint(breakpoint)

                if guidance:
                    # Emit resolution event
                    self.event_bridge.notify(
                        event_type="breakpoint_resolved",
                        breakpoint_id=breakpoint.breakpoint_id,
                        action=guidance.action,
                        reasoning=guidance.reasoning,
                    )

                    # Handle abort action
                    if guidance.action == "abort":
                        raise RuntimeError(f"Debate aborted by human: {guidance.reasoning}")

                    return guidance.reasoning or guidance.decision

        except ImportError:
            logger.debug("Breakpoints module not available")
        except Exception as e:
            logger.warning(f"Breakpoint check failed: {e}")

        return None

    def _build_proposal_prompt(self, agent: Agent) -> str:
        """Build the initial proposal prompt."""
        # Drain pending audience events before building prompt
        self._drain_user_events()

        # Sync state to PromptBuilder
        self._sync_prompt_builder_state()

        # Compute audience section (needs spectator callback)
        audience_section = ""
        if (
            self.protocol.audience_injection in ("summary", "inject")
            and self.user_suggestions
        ):
            clusters = cluster_suggestions(list(self.user_suggestions))
            audience_section = format_for_prompt(clusters)

            # Emit stream event for dashboard
            if self.spectator and clusters:
                self._notify_spectator(
                    "audience_summary",
                    details=f"{sum(c.count for c in clusters)} suggestions in {len(clusters)} clusters",
                    metric=len(clusters),
                )

        return self.prompt_builder.build_proposal_prompt(agent, audience_section)

    def _build_revision_prompt(
        self, agent: Agent, original: str, critiques: list[Critique]
    ) -> str:
        """Build the revision prompt including critiques."""
        # Drain pending audience events before building prompt
        self._drain_user_events()

        # Sync state to PromptBuilder
        self._sync_prompt_builder_state()

        # Compute audience section
        audience_section = ""
        if (
            self.protocol.audience_injection in ("summary", "inject")
            and self.user_suggestions
        ):
            clusters = cluster_suggestions(list(self.user_suggestions))
            audience_section = format_for_prompt(clusters)

        return self.prompt_builder.build_revision_prompt(
            agent, original, critiques, audience_section
        )

    def _build_judge_prompt(
        self, proposals: dict[str, str], task: str, critiques: list[Critique]
    ) -> str:
        """Build the judge/synthesizer prompt."""
        return self.prompt_builder.build_judge_prompt(proposals, task, critiques)
