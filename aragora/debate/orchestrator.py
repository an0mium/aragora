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
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.arena_initializer import ArenaInitializer
from aragora.debate.arena_phases import create_phase_executor, init_phases
from aragora.debate.audience_manager import AudienceManager
from aragora.debate.checkpoint_ops import CheckpointOperations
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
from aragora.debate.event_emission import EventEmitter
from aragora.debate.lifecycle_manager import LifecycleManager
from aragora.debate.grounded_operations import GroundedOperations
from aragora.debate.prompt_context import PromptContextBuilder
from aragora.debate.event_bus import EventBus
from aragora.debate.judge_selector import JudgeSelector
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.debate.result_formatter import ResultFormatter
from aragora.debate.roles_manager import RolesManager
from aragora.debate.sanitization import OutputSanitizer
from aragora.debate.state_cache import DebateStateCache
from aragora.debate.termination_checker import TerminationChecker
from aragora.exceptions import EarlyStopError
from aragora.observability.logging import correlation_context
from aragora.observability.logging import get_logger as get_structured_logger
from aragora.observability.tracing import add_span_attributes, get_tracer
from aragora.server.metrics import (
    ACTIVE_DEBATES,
    track_debate_outcome,
)
from aragora.spectate.stream import SpectatorStream
from aragora.utils.cache_registry import register_lru_cache

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
        hook_manager=None,  # Optional HookManager for extended lifecycle hooks
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
        knowledge_mound=None,  # Optional KnowledgeMound for unified knowledge queries/ingestion
        enable_knowledge_retrieval: bool = True,  # Query mound before debates
        enable_knowledge_ingestion: bool = True,  # Store consensus outcomes in mound
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
        # ML Integration (local ML models for routing, quality, consensus)
        enable_ml_delegation: bool = False,  # Use ML-based agent selection
        ml_delegation_strategy=None,  # Optional custom MLDelegationStrategy
        ml_delegation_weight: float = 0.3,  # Weight for ML scoring vs ELO (0.0-1.0)
        enable_quality_gates: bool = False,  # Filter low-quality responses via QualityGate
        quality_gate_threshold: float = 0.6,  # Minimum quality score (0.0-1.0)
        enable_consensus_estimation: bool = False,  # Use ConsensusEstimator for early termination
        consensus_early_termination_threshold: float = 0.85,  # Probability threshold
        # RLM Cognitive Load Limiter (for long debates)
        use_rlm_limiter: bool = False,  # Use RLM-enhanced cognitive limiter for context compression
        rlm_limiter=None,  # Pre-configured RLMCognitiveLoadLimiter
        rlm_compression_threshold: int = 3000,  # Chars above which to trigger RLM compression
        rlm_max_recent_messages: int = 5,  # Keep N most recent messages at full detail
        rlm_summary_level: str = "SUMMARY",  # Abstraction level for older content
    ):
        """Initialize the Arena with environment, agents, and optional subsystems.

        See inline parameter comments for subsystem descriptions.
        Initialization delegates to ArenaInitializer for core/tracker setup.
        """
        # Create initializer with broadcast callback
        initializer = ArenaInitializer(broadcast_callback=self._broadcast_health_event)

        # Initialize core configuration via ArenaInitializer
        core = initializer.init_core(
            environment=environment,
            agents=agents,
            protocol=protocol,
            memory=memory,
            event_hooks=event_hooks,
            hook_manager=hook_manager,
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
            enable_ml_delegation=enable_ml_delegation,
            ml_delegation_strategy=ml_delegation_strategy,
            ml_delegation_weight=ml_delegation_weight,
            enable_quality_gates=enable_quality_gates,
            quality_gate_threshold=quality_gate_threshold,
            enable_consensus_estimation=enable_consensus_estimation,
            consensus_early_termination_threshold=consensus_early_termination_threshold,
        )

        # Unpack core components to instance attributes
        self._apply_core_components(core)

        # Initialize tracking subsystems via ArenaInitializer
        trackers = initializer.init_trackers(
            protocol=self.protocol,
            loop_id=self.loop_id,
            agent_pool=self.agent_pool,
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
            knowledge_mound=knowledge_mound,
            enable_knowledge_retrieval=enable_knowledge_retrieval,
            enable_knowledge_ingestion=enable_knowledge_ingestion,
        )

        # Unpack tracker components to instance attributes
        self._apply_tracker_components(trackers)

        # Initialize user participation and roles
        self._init_user_participation()
        self._init_event_bus()
        self._init_roles_and_stances()

        # Initialize convergence detection and caches
        self._init_convergence()
        self._init_caches()

        # Initialize extracted helper classes for lifecycle, events, and checkpoints
        self._init_lifecycle_manager()
        self._init_event_emitter()
        self._init_checkpoint_ops()

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

        # Initialize RLM cognitive load limiter for context compression
        self._init_rlm_limiter(
            use_rlm_limiter=use_rlm_limiter,
            rlm_limiter=rlm_limiter,
            rlm_compression_threshold=rlm_compression_threshold,
            rlm_max_recent_messages=rlm_max_recent_messages,
            rlm_summary_level=rlm_summary_level,
        )

    @classmethod
    def from_config(
        cls,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol = None,
        config: ArenaConfig = None,
    ) -> "Arena":
        """Create an Arena from an ArenaConfig for cleaner dependency injection."""
        config = config or ArenaConfig()
        return cls(
            environment=environment,
            agents=agents,
            protocol=protocol,
            **config.to_arena_kwargs(),
        )

    def _apply_core_components(self, core) -> None:
        """Unpack CoreComponents dataclass to instance attributes."""
        self.env = core.env
        self.agents = core.agents
        self.protocol = core.protocol
        self.memory = core.memory
        self.hooks = core.hooks
        self.hook_manager = core.hook_manager
        self.event_emitter = core.event_emitter
        self.spectator = core.spectator
        self.debate_embeddings = core.debate_embeddings
        self.insight_store = core.insight_store
        self.recorder = core.recorder
        self.agent_weights = core.agent_weights
        self.loop_id = core.loop_id
        self.strict_loop_scoping = core.strict_loop_scoping
        self.circuit_breaker = core.circuit_breaker
        self.agent_pool = core.agent_pool
        self.immune_system = core.immune_system
        self.chaos_director = core.chaos_director
        self.performance_monitor = core.performance_monitor
        self.prompt_evolver = core.prompt_evolver
        self.autonomic = core.autonomic
        self.initial_messages = core.initial_messages
        self.trending_topic = core.trending_topic
        self.pulse_manager = core.pulse_manager
        self.auto_fetch_trending = core.auto_fetch_trending
        self.population_manager = core.population_manager
        self.auto_evolve = core.auto_evolve
        self.breeding_threshold = core.breeding_threshold
        self.evidence_collector = core.evidence_collector
        self.breakpoint_manager = core.breakpoint_manager
        self.agent_selector = core.agent_selector
        self.use_performance_selection = core.use_performance_selection
        self.checkpoint_manager = core.checkpoint_manager
        self.org_id = core.org_id
        self.user_id = core.user_id
        self.extensions = core.extensions
        self.cartographer = core.cartographer
        self.event_bridge = core.event_bridge
        # ML Integration
        self.enable_ml_delegation = core.enable_ml_delegation
        self.ml_delegation_weight = core.ml_delegation_weight
        self.enable_quality_gates = core.enable_quality_gates
        self.quality_gate_threshold = core.quality_gate_threshold
        self.enable_consensus_estimation = core.enable_consensus_estimation
        self.consensus_early_termination_threshold = core.consensus_early_termination_threshold
        self._ml_delegation_strategy = core.ml_delegation_strategy
        self._ml_quality_gate = core.ml_quality_gate
        self._ml_consensus_estimator = core.ml_consensus_estimator
        # Event bus initialized later in _init_event_bus() after audience_manager exists
        self.event_bus: Optional[EventBus] = None

    def _apply_tracker_components(self, trackers) -> None:
        """Unpack TrackerComponents dataclass to instance attributes."""
        self.position_tracker = trackers.position_tracker
        self.position_ledger = trackers.position_ledger
        self.elo_system = trackers.elo_system
        self.persona_manager = trackers.persona_manager
        self.dissent_retriever = trackers.dissent_retriever
        self.consensus_memory = trackers.consensus_memory
        self.flip_detector = trackers.flip_detector
        self.calibration_tracker = trackers.calibration_tracker
        self.continuum_memory = trackers.continuum_memory
        self.relationship_tracker = trackers.relationship_tracker
        self.moment_detector = trackers.moment_detector
        self.tier_analytics_tracker = trackers.tier_analytics_tracker
        self.knowledge_mound = trackers.knowledge_mound
        self.enable_knowledge_retrieval = trackers.enable_knowledge_retrieval
        self.enable_knowledge_ingestion = trackers.enable_knowledge_ingestion
        self._trackers = trackers.coordinator

    def _broadcast_health_event(self, event: dict) -> None:
        """Broadcast health events. Delegates to EventEmitter."""
        self._event_emitter.broadcast_health_event(event)

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
        """Initialize convergence detection if enabled."""
        self.convergence_detector = None
        self._convergence_debate_id = debate_id
        if self.protocol.convergence_detection:
            self.convergence_detector = ConvergenceDetector(
                convergence_threshold=self.protocol.convergence_threshold,
                divergence_threshold=self.protocol.divergence_threshold,
                min_rounds_before_check=1,
                debate_id=debate_id,
            )
        self._previous_round_responses: dict[str, str] = {}

    def _reinit_convergence_for_debate(self, debate_id: str) -> None:
        """Reinitialize convergence detector with debate-specific cache."""
        if self._convergence_debate_id == debate_id:
            return
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
        """Cleanup embedding cache for the current debate."""
        if self._convergence_debate_id:
            cleanup_embedding_cache(self._convergence_debate_id)
            logger.debug(f"Cleaned up embedding cache for debate {self._convergence_debate_id}")

    def _init_caches(self) -> None:
        """Initialize caches for computed values.

        Uses DebateStateCache to centralize all per-debate cached values.
        """
        self._cache = DebateStateCache()

    def _init_lifecycle_manager(self) -> None:
        """Initialize LifecycleManager for cleanup and task cancellation."""
        self._lifecycle = LifecycleManager(
            cache=self._cache,
            circuit_breaker=self.circuit_breaker,
            checkpoint_manager=self.checkpoint_manager,
        )

    def _init_event_emitter(self) -> None:
        """Initialize EventEmitter for spectator/websocket events."""
        self._event_emitter = EventEmitter(
            event_bus=self.event_bus,
            event_bridge=self.event_bridge,
            hooks=self.hooks,
            persona_manager=self.persona_manager,
        )

    def _init_checkpoint_ops(self) -> None:
        """Initialize CheckpointOperations for checkpoint and memory operations."""
        self._checkpoint_ops = CheckpointOperations(
            checkpoint_manager=self.checkpoint_manager,
            memory_manager=None,  # Set after _init_phases when memory_manager exists
            cache=self._cache,
        )

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

        # Set memory_manager on _checkpoint_ops now that it exists
        if hasattr(self, "_checkpoint_ops") and self._checkpoint_ops:
            self._checkpoint_ops.memory_manager = self.memory_manager

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

    def _init_rlm_limiter(
        self,
        use_rlm_limiter: bool,
        rlm_limiter,
        rlm_compression_threshold: int,
        rlm_max_recent_messages: int,
        rlm_summary_level: str,
    ) -> None:
        """Initialize the RLM cognitive load limiter for context compression.

        The RLM limiter compresses older debate context using hierarchical
        summarization while preserving semantic access. This prevents context
        windows from overflowing during long debates.
        """
        self.use_rlm_limiter = use_rlm_limiter
        self.rlm_compression_threshold = rlm_compression_threshold
        self.rlm_max_recent_messages = rlm_max_recent_messages
        self.rlm_summary_level = rlm_summary_level

        if rlm_limiter is not None:
            self.rlm_limiter = rlm_limiter
        elif use_rlm_limiter:
            # Create RLM limiter with configured parameters
            try:
                from aragora.debate.cognitive_limiter_rlm import (
                    RLMCognitiveBudget,
                    RLMCognitiveLoadLimiter,
                )

                budget = RLMCognitiveBudget(
                    enable_rlm_compression=True,
                    compression_threshold=rlm_compression_threshold,
                    max_recent_full_messages=rlm_max_recent_messages,
                    summary_level=rlm_summary_level,
                )
                self.rlm_limiter = RLMCognitiveLoadLimiter(budget=budget)
                logger.info(
                    f"[arena] RLM limiter enabled: threshold={rlm_compression_threshold}, "
                    f"recent={rlm_max_recent_messages}, level={rlm_summary_level}"
                )
            except ImportError:
                logger.warning("[arena] RLM module not available, disabling limiter")
                self.rlm_limiter = None
                self.use_rlm_limiter = False
        else:
            self.rlm_limiter = None

    def _require_agents(self) -> list[Agent]:
        """Return agents list, raising error if empty."""
        if not self.agents:
            raise ValueError("No agents available - Arena requires at least one agent")
        return self.agents

    def _sync_prompt_builder_state(self) -> None:
        """Sync Arena state to PromptBuilder before building prompts."""
        self.prompt_builder.current_role_assignments = self.current_role_assignments
        self.prompt_builder._historical_context_cache = self._cache.historical_context
        self.prompt_builder._continuum_context_cache = self._get_continuum_context()
        self.prompt_builder.user_suggestions = self.user_suggestions  # type: ignore[assignment]

    async def compress_debate_messages(
        self,
        messages: list,
        critiques: list | None = None,
    ) -> tuple[list, list | None]:
        """Compress debate messages using RLM cognitive load limiter.

        Uses hierarchical compression to reduce context size while preserving
        semantic content. Older messages are summarized, recent messages kept
        at full detail.

        Args:
            messages: List of debate messages to compress
            critiques: Optional list of critiques to compress

        Returns:
            Tuple of (compressed_messages, compressed_critiques)

        Example:
            compressed_msgs, compressed_crits = await arena.compress_debate_messages(
                messages=ctx.context_messages,
                critiques=all_critiques,
            )
        """
        if not self.use_rlm_limiter or not self.rlm_limiter:
            return messages, critiques

        try:
            result = await self.rlm_limiter.compress_context_async(
                messages=messages,
                critiques=critiques,
            )

            if result.compression_applied:
                logger.info(
                    f"[arena] Compressed debate context: {result.original_chars} → "
                    f"{result.compressed_chars} chars ({result.compression_ratio:.0%} of original)"
                )

            return result.messages, result.critiques
        except Exception as e:
            logger.warning(f"[arena] RLM compression failed, using original: {e}")
            return messages, critiques

    def _get_continuum_context(self) -> str:
        """Retrieve relevant memories from ContinuumMemory for debate context."""
        return self._context_delegator.get_continuum_context()

    def _store_debate_outcome_as_memory(self, result: "DebateResult") -> None:
        """Store debate outcome. Delegates to CheckpointOperations."""
        belief_cruxes = getattr(result, "belief_cruxes", None)
        if belief_cruxes:
            belief_cruxes = [str(c) for c in belief_cruxes[:10]]
        self._checkpoint_ops.store_debate_outcome(
            result, self.env.task, belief_cruxes=belief_cruxes
        )

    def _store_evidence_in_memory(self, evidence_snippets: list, task: str) -> None:
        """Store evidence. Delegates to CheckpointOperations."""
        self._checkpoint_ops.store_evidence(evidence_snippets, task)

    def _update_continuum_memory_outcomes(self, result: "DebateResult") -> None:
        """Update memory outcomes. Delegates to CheckpointOperations."""
        self._checkpoint_ops.update_memory_outcomes(result)

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
        """Select debate team using ML delegation or AgentPool.

        Priority:
        1. ML delegation (if enable_ml_delegation=True)
        2. Performance selection via AgentPool (if use_performance_selection=True)
        3. Original requested agents
        """
        # ML-based agent selection takes priority
        if self.enable_ml_delegation and self._ml_delegation_strategy:
            try:
                selected = self._ml_delegation_strategy.select_agents(
                    task=self.env.task,
                    agents=requested_agents,
                    context={
                        "domain": self._extract_debate_domain(),
                        "protocol": self.protocol,
                    },
                    max_agents=len(requested_agents),
                )
                logger.debug(
                    f"[ml] Selected {len(selected)} agents via ML delegation: "
                    f"{[a.name for a in selected]}"
                )
                return selected
            except Exception as e:
                logger.warning(f"[ml] ML delegation failed, falling back: {e}")

        # Fall back to performance-based selection
        if self.use_performance_selection:
            return self.agent_pool.select_team(
                domain=self._extract_debate_domain(),
                team_size=len(requested_agents),
            )

        return requested_agents

    def _filter_responses_by_quality(
        self, responses: list[tuple[str, str]], context: str = ""
    ) -> list[tuple[str, str]]:
        """Filter responses using ML quality gate if enabled.

        Args:
            responses: List of (agent_name, response_text) tuples
            context: Optional task context for quality assessment

        Returns:
            Filtered list containing only high-quality responses
        """
        if not self.enable_quality_gates or not self._ml_quality_gate:
            return responses

        try:
            filtered = self._ml_quality_gate.filter_responses(
                responses, context=context or self.env.task
            )
            removed = len(responses) - len(filtered)
            if removed > 0:
                logger.debug(
                    f"[ml] Quality gate filtered {removed} low-quality responses"
                )
            return filtered
        except Exception as e:
            logger.warning(f"[ml] Quality gate failed, keeping all responses: {e}")
            return responses

    def _should_terminate_early(
        self, responses: list[tuple[str, str]], current_round: int
    ) -> bool:
        """Check if debate should terminate early based on consensus estimation.

        Args:
            responses: List of (agent_name, response_text) tuples
            current_round: Current debate round number

        Returns:
            True if consensus is highly likely and safe to terminate early
        """
        if not self.enable_consensus_estimation or not self._ml_consensus_estimator:
            return False

        try:
            should_stop = self._ml_consensus_estimator.should_terminate_early(
                responses=responses,
                current_round=current_round,
                total_rounds=self.protocol.rounds,
                context=self.env.task,
            )
            if should_stop:
                logger.info(
                    f"[ml] Consensus estimator recommends early termination at round "
                    f"{current_round}/{self.protocol.rounds}"
                )
            return should_stop
        except Exception as e:
            logger.warning(f"[ml] Consensus estimation failed: {e}")
            return False

    def _get_calibration_weight(self, agent_name: str) -> float:
        """Get calibration weight. Delegates to AgentPool."""
        return self.agent_pool._get_calibration_weight(agent_name)

    def _compute_composite_judge_score(self, agent_name: str) -> float:
        """Compute composite judge score. Delegates to AgentPool."""
        return self.agent_pool._compute_composite_score(agent_name, self._extract_debate_domain())

    def _select_critics_for_proposal(
        self, proposal_agent: str, all_critics: list[Agent]
    ) -> list[Agent]:
        """Select critics for proposal. Delegates to AgentPool."""
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
        """Handle user participation events. Delegates to AudienceManager."""
        self.audience_manager.handle_event(event)

    def _drain_user_events(self) -> None:
        """Drain pending user events. Delegates to AudienceManager."""
        self.audience_manager.drain_events()

    def _notify_spectator(self, event_type: str, **kwargs) -> None:
        """Notify spectator. Delegates to EventEmitter."""
        self._event_emitter.notify_spectator(event_type, **kwargs)

    def _emit_moment_event(self, moment) -> None:
        """Emit moment event. Delegates to EventEmitter."""
        self._event_emitter.emit_moment(moment)

    def _emit_agent_preview(self) -> None:
        """Emit agent preview. Delegates to EventEmitter."""
        self._event_emitter.emit_agent_preview(self.agents, self.current_role_assignments)

    def _record_grounded_position(
        self,
        agent_name: str,
        content: str,
        debate_id: str,
        round_num: int,
        confidence: float = 0.7,
        domain: Optional[str] = None,
    ):
        """Record position. Delegates to GroundedOperations."""
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
        """Update relationships. Delegates to GroundedOperations."""
        self._grounded_ops.update_relationships(debate_id, participants, winner, votes)

    def _create_grounded_verdict(self, result: "DebateResult") -> Any:
        """Create verdict. Delegates to GroundedOperations."""
        return self._grounded_ops.create_grounded_verdict(result)

    async def _verify_claims_formally(self, result: "DebateResult") -> None:
        """Verify claims with Z3. Delegates to GroundedOperations."""
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

    async def _fetch_knowledge_context(self, task: str, limit: int = 10) -> Optional[str]:
        """Fetch relevant knowledge from Knowledge Mound for debate context.

        Queries the unified knowledge superstructure for semantically related
        knowledge items to inform the debate.

        Args:
            task: The debate task to find relevant knowledge for
            limit: Maximum number of knowledge items to retrieve

        Returns:
            Formatted string with knowledge context, or None if unavailable
        """
        if not self.knowledge_mound or not self.enable_knowledge_retrieval:
            return None

        try:
            # Query mound for semantically related knowledge
            results = await self.knowledge_mound.query_semantic(
                query=task,
                limit=limit,
                min_confidence=0.5,
            )

            if not results or not results.items:
                return None

            # Format knowledge for agent context
            lines = ["## KNOWLEDGE MOUND CONTEXT"]
            lines.append("Relevant knowledge from organizational memory:\n")

            for item in results.items[:limit]:
                source = getattr(item, "source", "unknown")
                confidence = getattr(item, "confidence", 0.0)
                content = getattr(item, "content", str(item))[:300]
                lines.append(f"**[{source}]** (confidence: {confidence:.0%})")
                lines.append(f"{content}")
                lines.append("")

            logger.info(f"  [knowledge_mound] Retrieved {len(results.items)} items for context")
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"  [knowledge_mound] Failed to fetch context: {e}")
            return None

    async def _ingest_debate_outcome(self, result: "DebateResult") -> None:
        """Store debate outcome in Knowledge Mound for future retrieval.

        Ingests the consensus conclusion and key claims from high-confidence
        debates into the organizational knowledge superstructure.

        Args:
            result: The debate result to ingest
        """
        if not self.knowledge_mound or not self.enable_knowledge_ingestion:
            return

        # Only ingest high-quality outcomes (consensus with decent confidence)
        if not result.final_answer or result.confidence < 0.5:
            logger.debug("  [knowledge_mound] Skipping low-confidence debate outcome")
            return

        try:
            from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

            # Build metadata from debate result
            metadata = {
                "debate_id": result.id,
                "task": self.env.task[:500] if self.env else "",
                "confidence": result.confidence,
                "consensus_reached": result.consensus_reached,
                "rounds_used": result.rounds_used,
                "participants": result.participants[:10] if result.participants else [],
                "winner": result.winner,
            }

            # Add belief cruxes if available
            if hasattr(result, "debate_cruxes") and result.debate_cruxes:
                metadata["crux_claims"] = [
                    str(c.get("claim", c))[:200] for c in result.debate_cruxes[:5]
                ]

            # Ingest the consensus conclusion
            ingestion_result = await self.knowledge_mound.store(
                IngestionRequest(
                    content=f"Debate Conclusion: {result.final_answer[:2000]}",
                    source_type=KnowledgeSource.DEBATE,
                    debate_id=result.id,
                    confidence=result.confidence,
                    workspace_id=self.knowledge_mound.workspace_id,
                    metadata=metadata,
                )
            )

            if ingestion_result.success:
                logger.info(
                    f"  [knowledge_mound] Ingested debate outcome (node_id={ingestion_result.node_id})"
                )

                # Emit event for dashboard
                self._notify_spectator(
                    "knowledge_ingested",
                    details="Stored debate conclusion in Knowledge Mound",
                    metric=result.confidence,
                )

        except Exception as e:
            logger.warning(f"  [knowledge_mound] Failed to ingest outcome: {e}")

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
        """Format debate conclusion. Delegates to ResultFormatter."""
        return ResultFormatter().format_conclusion(result)

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
        """Create checkpoint. Delegates to CheckpointOperations."""
        await self._checkpoint_ops.create_checkpoint(
            ctx, round_num, self.env, self.agents, self.protocol
        )

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

    def _track_circuit_breaker_metrics(self) -> None:
        """Track circuit breaker state in metrics. Delegates to LifecycleManager."""
        self._lifecycle.track_circuit_breaker_metrics()

    def _log_phase_failures(self, execution_result) -> None:
        """Log any failed phases. Delegates to LifecycleManager."""
        self._lifecycle.log_phase_failures(execution_result)

    async def _cleanup(self) -> None:
        """Internal cleanup. Delegates to LifecycleManager."""
        await self._lifecycle.cleanup()

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
            hook_manager=self.hook_manager,
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
