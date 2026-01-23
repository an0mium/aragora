"""
Arena configuration dataclass.

Extracted from orchestrator.py for modularity.
Provides type-safe configuration for Arena initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aragora.debate.protocol import CircuitBreaker
from aragora.spectate.stream import SpectatorStream
from aragora.type_protocols import (
    BroadcastPipelineProtocol,
    CalibrationTrackerProtocol,
    ConsensusMemoryProtocol,
    ContinuumMemoryProtocol,
    DebateEmbeddingsProtocol,
    DissentRetrieverProtocol,
    EloSystemProtocol,
    EventEmitterProtocol,
    EvidenceCollectorProtocol,
    FlipDetectorProtocol,
    InsightStoreProtocol,
    MomentDetectorProtocol,
    PersonaManagerProtocol,
    PopulationManagerProtocol,
    PositionLedgerProtocol,
    PositionTrackerProtocol,
    PromptEvolverProtocol,
    PulseManagerProtocol,
    RelationshipTrackerProtocol,
)


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
    memory: Optional[Any] = None  # CritiqueStore
    event_hooks: Optional[Dict[str, Any]] = None
    hook_manager: Optional[Any] = None  # HookManager for extended lifecycle hooks
    event_emitter: Optional[EventEmitterProtocol] = None
    spectator: Optional[SpectatorStream] = None
    debate_embeddings: Optional[DebateEmbeddingsProtocol] = None
    insight_store: Optional[InsightStoreProtocol] = None
    recorder: Optional[Any] = None  # ReplayRecorder
    circuit_breaker: Optional[CircuitBreaker] = None
    evidence_collector: Optional[EvidenceCollectorProtocol] = None

    # Agent configuration
    agent_weights: Optional[Dict[str, float]] = None

    # Vertical personas (industry-specific configuration)
    vertical: Optional[str] = None  # Industry vertical: "software", "legal", "healthcare", etc.
    vertical_persona_manager: Optional[Any] = None  # VerticalPersonaManager instance
    auto_detect_vertical: bool = True  # Auto-detect vertical from task description

    # Tracking subsystems
    position_tracker: Optional[PositionTrackerProtocol] = None
    position_ledger: Optional[PositionLedgerProtocol] = None
    enable_position_ledger: bool = False  # Auto-create PositionLedger if not provided
    elo_system: Optional[EloSystemProtocol] = None
    persona_manager: Optional[PersonaManagerProtocol] = None
    dissent_retriever: Optional[DissentRetrieverProtocol] = None
    consensus_memory: Optional[ConsensusMemoryProtocol] = None
    flip_detector: Optional[FlipDetectorProtocol] = None
    calibration_tracker: Optional[CalibrationTrackerProtocol] = None
    continuum_memory: Optional[ContinuumMemoryProtocol] = None
    relationship_tracker: Optional[RelationshipTrackerProtocol] = None
    moment_detector: Optional[MomentDetectorProtocol] = None
    tier_analytics_tracker: Optional[Any] = None  # TierAnalyticsTracker for memory ROI

    # Knowledge Mound integration
    knowledge_mound: Optional[Any] = None  # KnowledgeMound for unified knowledge queries/ingestion
    enable_knowledge_retrieval: bool = True  # Query mound before debates for relevant knowledge
    enable_knowledge_ingestion: bool = True  # Store consensus outcomes in mound after debates

    # Automatic knowledge revalidation (staleness detection)
    enable_auto_revalidation: bool = False  # Auto-trigger revalidation for stale knowledge
    revalidation_staleness_threshold: float = 0.7  # Staleness score threshold (0.0-1.0)
    revalidation_check_interval_seconds: int = 3600  # Interval between staleness checks (1 hour)
    revalidation_scheduler: Optional[Any] = None  # Pre-configured RevalidationScheduler

    # Belief Network guidance (cross-debate crux injection)
    enable_belief_guidance: bool = True  # Inject historical cruxes from similar debates as context

    # Cross-debate institutional memory
    cross_debate_memory: Optional[Any] = None  # CrossDebateMemory for institutional knowledge
    enable_cross_debate_memory: bool = True  # Inject institutional knowledge from past debates

    # Post-debate workflow automation
    post_debate_workflow: Optional[Any] = (
        None  # Workflow DAG to trigger after high-confidence debates
    )
    enable_post_debate_workflow: bool = False  # Auto-trigger workflow after debates
    post_debate_workflow_threshold: float = 0.7  # Min confidence to trigger workflow

    # Genesis evolution
    population_manager: Optional[PopulationManagerProtocol] = None
    auto_evolve: bool = False  # Trigger evolution after high-quality debates
    breeding_threshold: float = 0.8  # Min confidence to trigger evolution

    # Fork/continuation support
    initial_messages: Optional[List[Any]] = None
    trending_topic: Optional[Any] = None  # TrendingTopic
    pulse_manager: Optional[PulseManagerProtocol] = None
    auto_fetch_trending: bool = False  # Auto-fetch trending topics if none provided

    # Human-in-the-loop breakpoints
    breakpoint_manager: Optional[Any] = None  # BreakpointManager

    # Debate checkpointing for resume support
    checkpoint_manager: Optional[Any] = None  # CheckpointManager for pause/resume
    enable_checkpointing: bool = (
        True  # Auto-create CheckpointManager if True (enables debate resume)
    )

    # Performance telemetry
    performance_monitor: Optional[Any] = None  # AgentPerformanceMonitor
    enable_performance_monitor: bool = True  # Auto-create PerformanceMonitor for timing metrics
    enable_telemetry: bool = False  # Enable Prometheus/Blackbox telemetry emission

    # Agent selection (performance-based team formation)
    agent_selector: Optional[Any] = None  # AgentSelector for performance-based selection
    use_performance_selection: bool = (
        True  # Enable ELO/calibration-based agent selection (default: on)
    )

    # Airlock resilience layer
    use_airlock: bool = False  # Wrap agents with AirlockProxy for timeout/fallback
    airlock_config: Optional[Any] = None  # AirlockConfig for customization

    # Prompt evolution for self-improvement
    prompt_evolver: Optional[PromptEvolverProtocol] = None
    enable_prompt_evolution: bool = False  # Auto-create PromptEvolver if True

    # Billing/usage tracking (multi-tenancy)
    org_id: str = ""  # Organization ID for multi-tenancy
    user_id: str = ""  # User ID for usage attribution
    usage_tracker: Optional[Any] = None  # UsageTracker for token usage

    # Broadcast auto-trigger for high-quality debates
    broadcast_pipeline: Optional[BroadcastPipelineProtocol] = None
    auto_broadcast: bool = False  # Auto-trigger broadcast after high-quality debates
    broadcast_min_confidence: float = 0.8  # Minimum confidence to trigger broadcast
    broadcast_platforms: Optional[List[str]] = None  # Platforms to publish to (default: ["rss"])

    # Training data export (Tinker integration)
    training_exporter: Optional[Any] = None  # TrainingDataExporter for auto-export
    auto_export_training: bool = False  # Auto-export training data after debates
    training_export_min_confidence: float = 0.75  # Min confidence to export as SFT
    training_export_path: str = ""  # Output path for training data (default: data/training/)

    # ML Integration (local ML models for routing, quality, consensus)
    enable_ml_delegation: bool = False  # Use ML-based agent selection (MLDelegationStrategy)
    ml_delegation_strategy: Optional[Any] = None  # Custom MLDelegationStrategy instance
    ml_delegation_weight: float = 0.3  # Weight for ML scoring vs ELO (0.0-1.0)
    enable_quality_gates: bool = False  # Filter low-quality responses via QualityGate
    quality_gate_threshold: float = 0.6  # Minimum quality score (0.0-1.0)
    enable_consensus_estimation: bool = False  # Use ConsensusEstimator for early termination
    consensus_early_termination_threshold: float = 0.85  # Probability threshold for early stop

    # RLM Cognitive Load Limiter (for long debates)
    use_rlm_limiter: bool = True  # Use RLM-enhanced cognitive limiter for context compression (auto-triggers after rlm_compression_round_threshold)
    rlm_limiter: Optional[Any] = None  # Pre-configured RLMCognitiveLoadLimiter
    rlm_compression_threshold: int = 3000  # Chars above which to trigger RLM compression
    rlm_max_recent_messages: int = 5  # Keep N most recent messages at full detail
    rlm_summary_level: str = (
        "SUMMARY"  # Abstraction level for older content (ABSTRACT, SUMMARY, DETAILED)
    )
    rlm_compression_round_threshold: int = 3  # Start auto-compression after this many rounds

    # Memory Coordination (cross-system atomic writes)
    enable_coordinated_writes: bool = True  # Use MemoryCoordinator for atomic multi-system writes
    memory_coordinator: Optional[Any] = None  # Pre-configured MemoryCoordinator
    coordinator_parallel_writes: bool = (
        False  # Execute memory writes in parallel (False = safer sequential)
    )
    coordinator_rollback_on_failure: bool = True  # Roll back successful writes if any fails
    coordinator_min_confidence_for_mound: float = 0.7  # Min confidence to write to KnowledgeMound

    # Performance → Selection Feedback Loop
    enable_performance_feedback: bool = True  # Adjust selection weights based on debate performance
    selection_feedback_loop: Optional[Any] = None  # Pre-configured SelectionFeedbackLoop
    feedback_loop_weight: float = 0.15  # Weight for feedback adjustments (0.0-1.0)
    feedback_loop_decay: float = 0.9  # Decay factor for old feedback
    feedback_loop_min_debates: int = 3  # Min debates before applying feedback

    # Hook System Activation (cross-subsystem event wiring)
    enable_hook_handlers: bool = True  # Register default hook handlers via HookHandlerRegistry
    hook_handler_registry: Optional[Any] = None  # Pre-configured HookHandlerRegistry

    # Performance → ELO Integration (cross-pollination)
    enable_performance_elo: bool = True  # Use performance metrics to modulate ELO K-factors
    performance_elo_integrator: Optional[Any] = None  # Pre-configured PerformanceEloIntegrator

    # Outcome → Memory Integration (cross-pollination)
    enable_outcome_memory: bool = True  # Promote memories used in successful debates
    outcome_memory_bridge: Optional[Any] = None  # Pre-configured OutcomeMemoryBridge
    outcome_memory_success_threshold: float = 0.7  # Min confidence for promotion
    outcome_memory_usage_threshold: int = 3  # Successful uses before promotion

    # Trickster Auto-Calibration (cross-pollination)
    enable_trickster_calibration: bool = True  # Auto-calibrate Trickster based on outcomes
    trickster_calibrator: Optional[Any] = None  # Pre-configured TricksterCalibrator
    trickster_calibration_min_samples: int = 20  # Min outcomes before calibrating
    trickster_calibration_interval: int = 50  # Debates between calibrations

    # Checkpoint Memory State (cross-pollination)
    checkpoint_include_memory: bool = True  # Include continuum memory state in checkpoints
    checkpoint_memory_max_entries: int = 100  # Max entries per tier in checkpoint snapshot
    checkpoint_memory_restore_mode: str = "replace"  # Restore mode: replace, keep, or merge

    # =============================================================================
    # Phase 9: Cross-Pollination Bridges (Session 5+)
    # These bridges connect previously isolated subsystems for self-improvement
    # =============================================================================

    # Performance → Agent Router Bridge
    # Routes agents based on performance metrics (latency, quality, consistency)
    enable_performance_router: bool = True  # Use performance metrics to inform routing
    performance_router_bridge: Optional[Any] = None  # Pre-configured PerformanceRouterBridge
    performance_router_latency_weight: float = 0.3  # Weight for latency in routing score
    performance_router_quality_weight: float = 0.4  # Weight for quality in routing score
    performance_router_consistency_weight: float = 0.3  # Weight for consistency

    # Outcome → Complexity Governor Bridge
    # Adjusts complexity budgets based on outcome patterns
    enable_outcome_complexity: bool = True  # Use outcomes to inform complexity governance
    outcome_complexity_bridge: Optional[Any] = None  # Pre-configured OutcomeComplexityBridge
    outcome_complexity_high_success_boost: float = 0.1  # Boost for high-success agents
    outcome_complexity_low_success_penalty: float = 0.15  # Penalty for low-success agents
    outcome_complexity_min_outcomes: int = 5  # Min outcomes before applying adjustments

    # Analytics → Team Selection Bridge
    # Uses analytics patterns to improve team composition
    enable_analytics_selection: bool = True  # Use analytics to inform team selection
    analytics_selection_bridge: Optional[Any] = None  # Pre-configured AnalyticsSelectionBridge
    analytics_selection_diversity_weight: float = 0.2  # Weight for cognitive diversity
    analytics_selection_synergy_weight: float = 0.3  # Weight for historical synergy

    # Novelty → Selection Feedback Bridge
    # Penalizes low-novelty agents in selection, rewards diverse thinkers
    enable_novelty_selection: bool = True  # Use novelty metrics for selection feedback
    novelty_selection_bridge: Optional[Any] = None  # Pre-configured NoveltySelectionBridge
    novelty_selection_low_penalty: float = 0.15  # Penalty for consistently low novelty
    novelty_selection_high_bonus: float = 0.1  # Bonus for consistently high novelty
    novelty_selection_min_proposals: int = 10  # Min proposals before applying adjustments
    novelty_selection_low_threshold: float = 0.3  # Below this = low novelty

    # Relationship → Bias Mitigation Bridge
    # Detects echo chambers and adjusts voting weights
    enable_relationship_bias: bool = True  # Use relationships to detect/mitigate bias
    relationship_bias_bridge: Optional[Any] = None  # Pre-configured RelationshipBiasBridge
    relationship_bias_alliance_threshold: float = (
        0.7  # Alliance score threshold for echo chamber risk
    )
    relationship_bias_agreement_threshold: float = 0.8  # Agreement rate for echo chamber detection
    relationship_bias_vote_penalty: float = 0.3  # Weight penalty for allied voter-votee pairs
    relationship_bias_min_debates: int = 5  # Min debates before considering relationship

    # RLM → Selection Feedback Bridge
    # Optimizes selection for agents efficient with compressed context
    enable_rlm_selection: bool = True  # Use RLM metrics for selection feedback
    rlm_selection_bridge: Optional[Any] = None  # Pre-configured RLMSelectionBridge
    rlm_selection_min_operations: int = 5  # Min RLM operations before applying boost
    rlm_selection_compression_weight: float = 0.15  # Weight for compression efficiency
    rlm_selection_query_weight: float = 0.15  # Weight for query efficiency
    rlm_selection_max_boost: float = 0.25  # Maximum selection boost from RLM efficiency

    # Calibration → Cost Optimizer Bridge
    # Selects cost-efficient agents based on calibration quality
    enable_calibration_cost: bool = True  # Use calibration to optimize costs
    calibration_cost_bridge: Optional[Any] = None  # Pre-configured CalibrationCostBridge
    calibration_cost_min_predictions: int = 20  # Min predictions before scoring
    calibration_cost_ece_threshold: float = 0.1  # ECE threshold for "well-calibrated"
    calibration_cost_overconfident_multiplier: float = 1.3  # Cost multiplier for overconfident
    calibration_cost_weight: float = 0.3  # Weight for cost in efficiency score

    # =============================================================================
    # Phase 10: Bidirectional Knowledge Mound Integration (Session 7+)
    # These integrations enable two-way data flow between KM and source systems
    # =============================================================================

    # Master switch for all bidirectional KM integrations
    enable_km_bidirectional: bool = True  # Enable all reverse flows from KM

    # ContinuumMemory ↔ KM bidirectional sync
    # KM validation improves memory tier placement and importance scores
    enable_km_continuum_sync: bool = True  # Enable KM → ContinuumMemory reverse flow
    km_continuum_adapter: Optional[Any] = None  # Pre-configured ContinuumAdapter
    km_continuum_min_confidence: float = 0.7  # Min KM confidence for tier changes
    km_continuum_promotion_threshold: float = 0.8  # Cross-debate utility for promotion
    km_continuum_demotion_threshold: float = 0.3  # KM confidence for demotion
    km_continuum_sync_batch_size: int = 50  # Batch size for validation syncs

    # ELO/Ranking ↔ KM bidirectional sync
    # KM patterns influence agent ELO adjustments
    enable_km_elo_sync: bool = True  # Enable KM → ELO reverse flow
    km_elo_bridge: Optional[Any] = None  # Pre-configured KMEloBridge
    km_elo_min_pattern_confidence: float = 0.7  # Min confidence for ELO adjustments
    km_elo_max_adjustment: float = 50.0  # Max ELO change per sync
    km_elo_sync_interval_hours: int = 24  # Interval between KM → ELO syncs

    # OutcomeTracker ↔ KM bidirectional sync
    # Debate outcomes validate/invalidate KM entries
    enable_km_outcome_validation: bool = True  # Enable Outcome → KM validation
    km_outcome_bridge: Optional[Any] = None  # Pre-configured KMOutcomeBridge
    km_outcome_success_boost: float = 0.1  # Confidence boost for successful outcomes
    km_outcome_failure_penalty: float = 0.05  # Confidence penalty for failed outcomes
    km_outcome_propagation_depth: int = 2  # How deep to propagate validation in graph

    # BeliefNetwork ↔ KM bidirectional sync
    # KM patterns improve belief detection thresholds
    enable_km_belief_sync: bool = True  # Enable KM → BeliefNetwork reverse flow
    km_belief_adapter: Optional[Any] = None  # Pre-configured BeliefAdapter
    km_belief_threshold_min_samples: int = 50  # Min samples before threshold updates
    km_belief_crux_sensitivity_range: tuple = (0.2, 0.8)  # Bounds for crux sensitivity

    # InsightStore/FlipDetector ↔ KM bidirectional sync
    # KM patterns improve flip detection baselines
    enable_km_flip_sync: bool = True  # Enable KM → FlipDetector reverse flow
    km_insights_adapter: Optional[Any] = None  # Pre-configured InsightsAdapter
    km_flip_min_outcomes: int = 20  # Min outcomes before baseline updates
    km_flip_sensitivity_range: tuple = (0.3, 0.9)  # Bounds for flip sensitivity

    # CritiqueStore ↔ KM bidirectional sync
    # KM validation boosts successful critique patterns
    enable_km_critique_sync: bool = True  # Enable KM → CritiqueStore reverse flow
    km_critique_adapter: Optional[Any] = None  # Pre-configured CritiqueAdapter
    km_critique_success_boost: float = 0.15  # Pattern score boost for success
    km_critique_min_validations: int = 5  # Min validations before boosting

    # Pulse/Trending ↔ KM bidirectional sync
    # KM coverage influences topic scheduling
    enable_km_pulse_sync: bool = True  # Enable KM → Pulse reverse flow
    km_pulse_adapter: Optional[Any] = None  # Pre-configured PulseAdapter
    km_pulse_coverage_weight: float = 0.2  # Weight for KM coverage in scheduling
    km_pulse_recommend_limit: int = 10  # Max topic recommendations per sync

    # Global bidirectional sync settings
    enable_km_coordinator: bool = True  # Auto-create BidirectionalCoordinator if KM available
    km_sync_interval_seconds: int = 300  # Interval between bidirectional syncs (5 min)
    km_min_confidence_for_reverse: float = 0.7  # Min confidence for any reverse flow
    km_parallel_sync: bool = True  # Run adapter syncs in parallel (faster, more resource intensive)
    km_bidirectional_coordinator: Optional[Any] = None  # Pre-configured BidirectionalCoordinator

    def __post_init__(self) -> None:
        """Initialize defaults that can't be set in field definitions."""
        if self.broadcast_platforms is None:
            self.broadcast_platforms = ["rss"]

    def to_arena_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs dict for Arena.__init__.

        Returns:
            Dictionary of keyword arguments for Arena initialization.

        Note:
            Only includes parameters that Arena.__init__ currently accepts.
            broadcast_platforms and training_export_path are stored in config
            but not yet supported by Arena.
        """
        return {
            "memory": self.memory,
            "event_hooks": self.event_hooks,
            "hook_manager": self.hook_manager,
            "event_emitter": self.event_emitter,
            "spectator": self.spectator,
            "debate_embeddings": self.debate_embeddings,
            "insight_store": self.insight_store,
            "recorder": self.recorder,
            "agent_weights": self.agent_weights,
            "position_tracker": self.position_tracker,
            "position_ledger": self.position_ledger,
            "enable_position_ledger": self.enable_position_ledger,
            "elo_system": self.elo_system,
            "persona_manager": self.persona_manager,
            "dissent_retriever": self.dissent_retriever,
            "consensus_memory": self.consensus_memory,
            "flip_detector": self.flip_detector,
            "calibration_tracker": self.calibration_tracker,
            "continuum_memory": self.continuum_memory,
            "relationship_tracker": self.relationship_tracker,
            "moment_detector": self.moment_detector,
            "tier_analytics_tracker": self.tier_analytics_tracker,
            "knowledge_mound": self.knowledge_mound,
            "enable_knowledge_retrieval": self.enable_knowledge_retrieval,
            "enable_knowledge_ingestion": self.enable_knowledge_ingestion,
            # Auto-revalidation (enable_auto_revalidation is passed, detailed config is stored in ArenaConfig)
            "enable_auto_revalidation": self.enable_auto_revalidation,
            # Note: revalidation_staleness_threshold, revalidation_check_interval_seconds,
            # and revalidation_scheduler are stored in config but Arena reads them from config
            "enable_belief_guidance": self.enable_belief_guidance,
            # Cross-debate institutional memory
            "cross_debate_memory": self.cross_debate_memory,
            "enable_cross_debate_memory": self.enable_cross_debate_memory,
            # Post-debate workflow automation
            "post_debate_workflow": self.post_debate_workflow,
            "enable_post_debate_workflow": self.enable_post_debate_workflow,
            "post_debate_workflow_threshold": self.post_debate_workflow_threshold,
            "loop_id": self.loop_id,
            "strict_loop_scoping": self.strict_loop_scoping,
            "circuit_breaker": self.circuit_breaker,
            "initial_messages": self.initial_messages,
            "trending_topic": self.trending_topic,
            "pulse_manager": self.pulse_manager,
            "auto_fetch_trending": self.auto_fetch_trending,
            "population_manager": self.population_manager,
            "auto_evolve": self.auto_evolve,
            "breeding_threshold": self.breeding_threshold,
            "evidence_collector": self.evidence_collector,
            "breakpoint_manager": self.breakpoint_manager,
            "performance_monitor": self.performance_monitor,
            "enable_performance_monitor": self.enable_performance_monitor,
            "enable_telemetry": self.enable_telemetry,
            "use_airlock": self.use_airlock,
            "airlock_config": self.airlock_config,
            "agent_selector": self.agent_selector,
            "use_performance_selection": self.use_performance_selection,
            "prompt_evolver": self.prompt_evolver,
            "enable_prompt_evolution": self.enable_prompt_evolution,
            "checkpoint_manager": self.checkpoint_manager,
            "enable_checkpointing": self.enable_checkpointing,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "usage_tracker": self.usage_tracker,
            "broadcast_pipeline": self.broadcast_pipeline,
            "auto_broadcast": self.auto_broadcast,
            "broadcast_min_confidence": self.broadcast_min_confidence,
            "training_exporter": self.training_exporter,
            "auto_export_training": self.auto_export_training,
            "training_export_min_confidence": self.training_export_min_confidence,
            # ML Integration
            "enable_ml_delegation": self.enable_ml_delegation,
            "ml_delegation_strategy": self.ml_delegation_strategy,
            "ml_delegation_weight": self.ml_delegation_weight,
            "enable_quality_gates": self.enable_quality_gates,
            "quality_gate_threshold": self.quality_gate_threshold,
            "enable_consensus_estimation": self.enable_consensus_estimation,
            "consensus_early_termination_threshold": self.consensus_early_termination_threshold,
            # RLM Cognitive Limiter
            "use_rlm_limiter": self.use_rlm_limiter,
            "rlm_limiter": self.rlm_limiter,
            "rlm_compression_threshold": self.rlm_compression_threshold,
            "rlm_max_recent_messages": self.rlm_max_recent_messages,
            "rlm_summary_level": self.rlm_summary_level,
            "rlm_compression_round_threshold": self.rlm_compression_round_threshold,
            # Note: The following are stored in ArenaConfig but not yet in Arena.__init__:
            # - Memory Coordination: enable_coordinated_writes, memory_coordinator,
            #   coordinator_parallel_writes, coordinator_rollback_on_failure,
            #   coordinator_min_confidence_for_mound
            # - Selection Feedback Loop: enable_performance_feedback, selection_feedback_loop,
            #   feedback_loop_weight, feedback_loop_decay, feedback_loop_min_debates
            # - Hook System: enable_hook_handlers, hook_handler_registry
            # - Broadcast: broadcast_platforms, training_export_path
            # - Phase 9 Cross-Pollination Bridges (auto-initialized by SubsystemCoordinator):
            #   * Performance Router: enable_performance_router, performance_router_bridge,
            #     performance_router_latency_weight, performance_router_quality_weight,
            #     performance_router_consistency_weight
            #   * Outcome Complexity: enable_outcome_complexity, outcome_complexity_bridge,
            #     outcome_complexity_high_success_boost, outcome_complexity_low_success_penalty,
            #     outcome_complexity_min_outcomes
            #   * Analytics Selection: enable_analytics_selection, analytics_selection_bridge,
            #     analytics_selection_diversity_weight, analytics_selection_synergy_weight
            #   * Novelty Selection: enable_novelty_selection, novelty_selection_bridge,
            #     novelty_selection_low_penalty, novelty_selection_high_bonus,
            #     novelty_selection_min_proposals, novelty_selection_low_threshold
            #   * Relationship Bias: enable_relationship_bias, relationship_bias_bridge,
            #     relationship_bias_alliance_threshold, relationship_bias_agreement_threshold,
            #     relationship_bias_vote_penalty, relationship_bias_min_debates
            #   * RLM Selection: enable_rlm_selection, rlm_selection_bridge,
            #     rlm_selection_min_operations, rlm_selection_compression_weight,
            #     rlm_selection_query_weight, rlm_selection_max_boost
            #   * Calibration Cost: enable_calibration_cost, calibration_cost_bridge,
            #     calibration_cost_min_predictions, calibration_cost_ece_threshold,
            #     calibration_cost_overconfident_multiplier, calibration_cost_weight
        }


__all__ = ["ArenaConfig"]
