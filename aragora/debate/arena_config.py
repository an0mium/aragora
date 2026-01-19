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
from aragora.typing import (
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
    enable_checkpointing: bool = False  # Auto-create CheckpointManager if True

    # Performance telemetry
    performance_monitor: Optional[Any] = None  # AgentPerformanceMonitor
    enable_performance_monitor: bool = True  # Auto-create PerformanceMonitor for timing metrics
    enable_telemetry: bool = False  # Enable Prometheus/Blackbox telemetry emission

    # Agent selection (performance-based team formation)
    agent_selector: Optional[Any] = None  # AgentSelector for performance-based selection
    use_performance_selection: bool = False  # Enable ELO/calibration-based agent selection

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

    def __post_init__(self) -> None:
        """Initialize defaults that can't be set in field definitions."""
        if self.broadcast_platforms is None:
            self.broadcast_platforms = ["rss"]

    def to_arena_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs dict for Arena.__init__.

        Returns:
            Dictionary of keyword arguments for Arena initialization.
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
        }


__all__ = ["ArenaConfig"]
