"""
Arena configuration dataclass.

Extracted from orchestrator.py for modularity.
Provides type-safe configuration for Arena initialization.

The configuration is organized into logical sub-config groups using the
strategy/builder pattern. Each group is a standalone dataclass that can
be used independently or composed into the main ArenaConfig.

Backward compatibility is preserved: all fields can still be accessed
directly on ArenaConfig (e.g., ``config.enable_consensus``) and all
fields can still be passed as flat kwargs to ``ArenaConfig()``.

Sub-config groups
-----------------
- **HookConfig**: Event hooks, YAML hooks, hook handlers
- **TrackingConfig**: Position, ELO, persona, flip detection, relationships
- **KnowledgeMoundConfig**: Knowledge retrieval, ingestion, extraction, revalidation
- **MemoryCoordinationConfig**: Cross-system atomic writes, rollback policy
- **PerformanceFeedbackConfig**: Selection feedback loop, performance-ELO integration
- **AuditTrailConfig**: Decision receipts, evidence provenance, bead tracking
- **MLIntegrationConfig**: ML delegation, quality gates, consensus estimation
- **RLMCognitiveConfig**: RLM cognitive load limiter settings
- **CheckpointMemoryConfig**: Checkpoint manager, memory state in checkpoints
- **CrossPollinationConfig**: Phase 9 cross-pollination bridges
- **KMBidirectionalConfig**: Phase 10 bidirectional Knowledge Mound sync
- **TranslationConfig**: Multi-language translation support

Builder usage
-------------
::

    config = (
        ArenaConfig.builder()
        .with_knowledge(enable_knowledge_retrieval=True)
        .with_audit_trail(enable_receipt_generation=True)
        .with_ml(enable_ml_delegation=True, ml_delegation_weight=0.5)
        .build()
    )
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import TYPE_CHECKING, Any, Optional

from aragora.config import DEFAULT_ROUNDS
from aragora.debate.protocol import CircuitBreaker

if TYPE_CHECKING:
    from aragora.debate.protocol import DebateProtocol
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


# =============================================================================
# Sub-Config Dataclasses
# =============================================================================


@dataclass
class HookConfig:
    """Event hooks and YAML hook configuration.

    Groups parameters related to event hooks, YAML-based declarative hooks,
    and the hook handler registry.

    Example::

        hook_cfg = HookConfig(
            enable_yaml_hooks=True,
            yaml_hooks_dir="custom_hooks",
        )
    """

    event_hooks: Optional[dict[str, Any]] = None
    hook_manager: Any | None = None  # HookManager for extended lifecycle hooks
    yaml_hooks_dir: str = "hooks"  # Directory to search for YAML hook definitions
    enable_yaml_hooks: bool = True  # Auto-discover and load YAML hooks on startup
    yaml_hooks_recursive: bool = True  # Search subdirectories for YAML hooks
    enable_hook_handlers: bool = True  # Register default hook handlers via HookHandlerRegistry
    hook_handler_registry: Any | None = None  # Pre-configured HookHandlerRegistry


@dataclass
class TrackingConfig:
    """Agent tracking subsystem configuration.

    Groups parameters for position tracking, ELO ratings, personas,
    flip detection, relationships, and moment detection.

    Example::

        tracking_cfg = TrackingConfig(
            enable_position_ledger=True,
            elo_system=my_elo,
        )
    """

    position_tracker: PositionTrackerProtocol | None = None
    position_ledger: PositionLedgerProtocol | None = None
    enable_position_ledger: bool = False  # Auto-create PositionLedger if not provided
    elo_system: EloSystemProtocol | None = None
    persona_manager: PersonaManagerProtocol | None = None
    dissent_retriever: DissentRetrieverProtocol | None = None
    consensus_memory: ConsensusMemoryProtocol | None = None
    flip_detector: FlipDetectorProtocol | None = None
    calibration_tracker: CalibrationTrackerProtocol | None = None
    continuum_memory: ContinuumMemoryProtocol | None = None
    relationship_tracker: RelationshipTrackerProtocol | None = None
    moment_detector: MomentDetectorProtocol | None = None
    tier_analytics_tracker: Any | None = None  # TierAnalyticsTracker for memory ROI


@dataclass
class KnowledgeMoundConfig:
    """Knowledge Mound integration configuration.

    Groups parameters for knowledge retrieval, ingestion, extraction,
    belief guidance, cross-debate memory, and auto-revalidation.

    Example::

        km_cfg = KnowledgeMoundConfig(
            enable_knowledge_retrieval=True,
            enable_knowledge_extraction=True,
            extraction_min_confidence=0.5,
        )
    """

    knowledge_mound: Any | None = None  # KnowledgeMound instance
    enable_knowledge_retrieval: bool = True  # Query mound before debates
    enable_knowledge_ingestion: bool = True  # Store consensus outcomes after debates
    enable_knowledge_extraction: bool = False  # Extract structured claims/relationships
    extraction_min_confidence: float = 0.3  # Min debate confidence to trigger extraction

    # Automatic knowledge revalidation (staleness detection)
    enable_auto_revalidation: bool = False  # Auto-trigger revalidation for stale knowledge
    revalidation_staleness_threshold: float = 0.7  # Staleness score threshold (0.0-1.0)
    revalidation_check_interval_seconds: int = 3600  # Interval between staleness checks
    revalidation_scheduler: Any | None = None  # Pre-configured RevalidationScheduler

    # Belief Network guidance (cross-debate crux injection)
    enable_belief_guidance: bool = True  # Inject historical cruxes as context

    # Cross-debate institutional memory
    cross_debate_memory: Any | None = None  # CrossDebateMemory for institutional knowledge
    enable_cross_debate_memory: bool = True  # Inject institutional knowledge from past debates


@dataclass
class MemoryCoordinationConfig:
    """Cross-system atomic memory write configuration.

    Groups parameters for the MemoryCoordinator which ensures atomic
    writes across multiple memory subsystems.

    Example::

        mc_cfg = MemoryCoordinationConfig(
            enable_coordinated_writes=True,
            coordinator_parallel_writes=True,
        )
    """

    enable_coordinated_writes: bool = True  # Use MemoryCoordinator for atomic writes
    memory_coordinator: Any | None = None  # Pre-configured MemoryCoordinator
    coordinator_parallel_writes: bool = False  # Execute writes in parallel (False = sequential)
    coordinator_rollback_on_failure: bool = True  # Roll back on partial failure
    coordinator_min_confidence_for_mound: float = 0.7  # Min confidence to write to KM


@dataclass
class PerformanceFeedbackConfig:
    """Performance feedback loop and ELO integration configuration.

    Groups parameters for the selection feedback loop that adjusts agent
    weights based on debate performance, and the performance-ELO integrator.

    Example::

        pf_cfg = PerformanceFeedbackConfig(
            enable_performance_feedback=True,
            feedback_loop_weight=0.2,
        )
    """

    # Performance -> Selection Feedback Loop
    enable_performance_feedback: bool = True  # Adjust selection weights from performance
    selection_feedback_loop: Any | None = None  # Pre-configured SelectionFeedbackLoop
    feedback_loop_weight: float = 0.15  # Weight for feedback adjustments (0.0-1.0)
    feedback_loop_decay: float = 0.9  # Decay factor for old feedback
    feedback_loop_min_debates: int = 3  # Min debates before applying feedback

    # Performance -> ELO Integration (cross-pollination)
    enable_performance_elo: bool = True  # Use performance metrics to modulate ELO K-factors
    performance_elo_integrator: Any | None = None  # Pre-configured PerformanceEloIntegrator

    # Outcome -> Memory Integration (cross-pollination)
    enable_outcome_memory: bool = True  # Promote memories used in successful debates
    outcome_memory_bridge: Any | None = None  # Pre-configured OutcomeMemoryBridge
    outcome_memory_success_threshold: float = 0.7  # Min confidence for promotion
    outcome_memory_usage_threshold: int = 3  # Successful uses before promotion

    # Trickster Auto-Calibration (cross-pollination)
    enable_trickster_calibration: bool = True  # Auto-calibrate Trickster based on outcomes
    trickster_calibrator: Any | None = None  # Pre-configured TricksterCalibrator
    trickster_calibration_min_samples: int = 20  # Min outcomes before calibrating
    trickster_calibration_interval: int = 50  # Debates between calibrations


@dataclass
class AuditTrailConfig:
    """Decision audit trail configuration.

    Groups parameters for decision receipts, evidence provenance tracking,
    and bead (git-backed) tracking.

    Example::

        audit_cfg = AuditTrailConfig(
            enable_receipt_generation=True,
            receipt_auto_sign=True,
            enable_provenance=True,
        )
    """

    # Decision Receipt Generation
    enable_receipt_generation: bool = False  # Auto-generate decision receipts
    receipt_min_confidence: float = 0.6  # Min confidence to generate receipt (0.0-1.0)
    receipt_auto_sign: bool = False  # Auto-sign receipts with HMAC-SHA256
    receipt_store: Any | None = None  # Pre-configured receipt store for persistence

    # Evidence Provenance Tracking (cryptographic audit trail)
    enable_provenance: bool = False  # Enable evidence provenance tracking
    provenance_manager: Any | None = None  # Pre-configured ProvenanceManager
    provenance_store: Any | None = None  # Pre-configured ProvenanceStore
    provenance_auto_persist: bool = True  # Auto-persist provenance chain after debate

    # Bead Tracking (git-backed audit trail)
    enable_bead_tracking: bool = False  # Create Bead for each debate decision
    bead_store: Any | None = None  # Pre-configured BeadStore for persistence
    bead_min_confidence: float = 0.5  # Min confidence to create a bead (0.0-1.0)
    bead_auto_commit: bool = False  # Auto-commit beads to git after creation


@dataclass
class MLIntegrationConfig:
    """ML integration configuration.

    Groups parameters for ML-based agent delegation, quality gates,
    and consensus estimation for early termination.

    Example::

        ml_cfg = MLIntegrationConfig(
            enable_ml_delegation=True,
            ml_delegation_weight=0.5,
            enable_quality_gates=True,
        )
    """

    enable_ml_delegation: bool = False  # Use ML-based agent selection
    ml_delegation_strategy: Any | None = None  # Custom MLDelegationStrategy
    ml_delegation_weight: float = 0.3  # Weight for ML scoring vs ELO (0.0-1.0)
    enable_quality_gates: bool = False  # Filter low-quality responses via QualityGate
    quality_gate_threshold: float = 0.6  # Minimum quality score (0.0-1.0)
    enable_consensus_estimation: bool = False  # Use ConsensusEstimator for early termination
    consensus_early_termination_threshold: float = 0.85  # Probability threshold for early stop


@dataclass
class RLMCognitiveConfig:
    """RLM cognitive load limiter configuration.

    Groups parameters for the RLM-enhanced cognitive limiter that manages
    context compression in long debates.

    Example::

        rlm_cfg = RLMCognitiveConfig(
            use_rlm_limiter=True,
            rlm_compression_threshold=2000,
        )
    """

    use_rlm_limiter: bool = True  # Use RLM-enhanced cognitive limiter
    rlm_limiter: Any | None = None  # Pre-configured RLMCognitiveLoadLimiter
    rlm_compression_threshold: int = 3000  # Chars above which to trigger compression
    rlm_max_recent_messages: int = 5  # Keep N most recent messages at full detail
    rlm_summary_level: str = "SUMMARY"  # Abstraction level (ABSTRACT, SUMMARY, DETAILED)
    rlm_compression_round_threshold: int = 3  # Start auto-compression after this many rounds


@dataclass
class CheckpointMemoryConfig:
    """Checkpoint and memory state configuration.

    Groups parameters for debate checkpointing (pause/resume) and
    the inclusion of memory state in checkpoint snapshots.

    Example::

        cp_cfg = CheckpointMemoryConfig(
            enable_checkpointing=True,
            checkpoint_include_memory=True,
        )
    """

    checkpoint_manager: Any | None = None  # CheckpointManager for pause/resume
    enable_checkpointing: bool = True  # Auto-create CheckpointManager
    checkpoint_include_memory: bool = True  # Include continuum memory in checkpoints
    checkpoint_memory_max_entries: int = 100  # Max entries per tier in snapshot
    checkpoint_memory_restore_mode: str = "replace"  # Restore mode: replace, keep, or merge


@dataclass
class CrossPollinationConfig:
    """Phase 9 cross-pollination bridge configuration.

    Groups parameters for bridges that connect previously isolated
    subsystems for self-improvement.

    Example::

        cp_cfg = CrossPollinationConfig(
            enable_performance_router=True,
            enable_novelty_selection=True,
        )
    """

    # Performance -> Agent Router Bridge
    enable_performance_router: bool = True
    performance_router_bridge: Any | None = None
    performance_router_latency_weight: float = 0.3
    performance_router_quality_weight: float = 0.4
    performance_router_consistency_weight: float = 0.3

    # Outcome -> Complexity Governor Bridge
    enable_outcome_complexity: bool = True
    outcome_complexity_bridge: Any | None = None
    outcome_complexity_high_success_boost: float = 0.1
    outcome_complexity_low_success_penalty: float = 0.15
    outcome_complexity_min_outcomes: int = 5

    # Analytics -> Team Selection Bridge
    enable_analytics_selection: bool = True
    analytics_selection_bridge: Any | None = None
    analytics_selection_diversity_weight: float = 0.2
    analytics_selection_synergy_weight: float = 0.3

    # Novelty -> Selection Feedback Bridge
    enable_novelty_selection: bool = True
    novelty_selection_bridge: Any | None = None
    novelty_selection_low_penalty: float = 0.15
    novelty_selection_high_bonus: float = 0.1
    novelty_selection_min_proposals: int = 10
    novelty_selection_low_threshold: float = 0.3

    # Relationship -> Bias Mitigation Bridge
    enable_relationship_bias: bool = True
    relationship_bias_bridge: Any | None = None
    relationship_bias_alliance_threshold: float = 0.7
    relationship_bias_agreement_threshold: float = 0.8
    relationship_bias_vote_penalty: float = 0.3
    relationship_bias_min_debates: int = 5

    # RLM -> Selection Feedback Bridge
    enable_rlm_selection: bool = True
    rlm_selection_bridge: Any | None = None
    rlm_selection_min_operations: int = 5
    rlm_selection_compression_weight: float = 0.15
    rlm_selection_query_weight: float = 0.15
    rlm_selection_max_boost: float = 0.25

    # Calibration -> Cost Optimizer Bridge
    enable_calibration_cost: bool = True
    calibration_cost_bridge: Any | None = None
    calibration_cost_min_predictions: int = 20
    calibration_cost_ece_threshold: float = 0.1
    calibration_cost_overconfident_multiplier: float = 1.3
    calibration_cost_weight: float = 0.3


@dataclass
class KMBidirectionalConfig:
    """Phase 10 bidirectional Knowledge Mound integration configuration.

    Groups parameters that enable two-way data flow between the Knowledge
    Mound and source systems (memory, ELO, outcomes, beliefs, etc.).

    Example::

        km_bi_cfg = KMBidirectionalConfig(
            enable_km_bidirectional=True,
            km_parallel_sync=True,
        )
    """

    # Master switch
    enable_km_bidirectional: bool = True

    # ContinuumMemory <-> KM
    enable_km_continuum_sync: bool = True
    km_continuum_adapter: Any | None = None
    km_continuum_min_confidence: float = 0.7
    km_continuum_promotion_threshold: float = 0.8
    km_continuum_demotion_threshold: float = 0.3
    km_continuum_sync_batch_size: int = 50

    # ELO/Ranking <-> KM
    enable_km_elo_sync: bool = True
    km_elo_bridge: Any | None = None
    km_elo_min_pattern_confidence: float = 0.7
    km_elo_max_adjustment: float = 50.0
    km_elo_sync_interval_hours: int = 24

    # OutcomeTracker <-> KM
    enable_km_outcome_validation: bool = True
    km_outcome_bridge: Any | None = None
    km_outcome_success_boost: float = 0.1
    km_outcome_failure_penalty: float = 0.05
    km_outcome_propagation_depth: int = 2

    # BeliefNetwork <-> KM
    enable_km_belief_sync: bool = True
    km_belief_adapter: Any | None = None
    km_belief_threshold_min_samples: int = 50
    km_belief_crux_sensitivity_range: tuple = (0.2, 0.8)

    # InsightStore/FlipDetector <-> KM
    enable_km_flip_sync: bool = True
    km_insights_adapter: Any | None = None
    km_flip_min_outcomes: int = 20
    km_flip_sensitivity_range: tuple = (0.3, 0.9)

    # CritiqueStore <-> KM
    enable_km_critique_sync: bool = True
    km_critique_adapter: Any | None = None
    km_critique_success_boost: float = 0.15
    km_critique_min_validations: int = 5

    # Pulse/Trending <-> KM
    enable_km_pulse_sync: bool = True
    km_pulse_adapter: Any | None = None
    km_pulse_coverage_weight: float = 0.2
    km_pulse_recommend_limit: int = 10

    # Global bidirectional sync settings
    enable_km_coordinator: bool = True
    km_sync_interval_seconds: int = 300
    km_min_confidence_for_reverse: float = 0.7
    km_parallel_sync: bool = True
    km_bidirectional_coordinator: Any | None = None


@dataclass
class TranslationSubConfig:
    """Multi-language translation support configuration.

    Groups parameters for translation services, language detection,
    and conclusion translation.

    Example::

        t_cfg = TranslationSubConfig(
            enable_translation=True,
            target_languages=["en", "fr", "de"],
        )
    """

    translation_service: Any | None = None  # Pre-configured TranslationService
    multilingual_manager: Any | None = None  # Pre-configured MultilingualDebateManager
    enable_translation: bool = False  # Enable multi-language debate support
    default_language: str = "en"  # Default language code (ISO 639-1)
    target_languages: Optional[list[str]] = None  # Languages to translate conclusions to
    auto_detect_language: bool = True  # Auto-detect source language of messages
    translate_conclusions: bool = True  # Translate final conclusions to target languages
    translation_cache_ttl_seconds: int = 3600  # Translation cache TTL (1 hour default)
    translation_cache_max_entries: int = 10000  # Max entries in translation cache


# =============================================================================
# Sub-config field name -> sub-config attribute name mapping
# Built once at module load for O(1) lookups.
# =============================================================================

_SUB_CONFIG_GROUPS: dict[str, tuple[str, type]] = {}
"""Maps field_name -> (sub_config_attr_on_ArenaConfig, sub_config_class)."""

_SUB_CONFIG_ATTRS: list[tuple[str, type]] = [
    ("hook_config", HookConfig),
    ("tracking_config", TrackingConfig),
    ("knowledge_config", KnowledgeMoundConfig),
    ("memory_coordination_config", MemoryCoordinationConfig),
    ("performance_feedback_config", PerformanceFeedbackConfig),
    ("audit_trail_config", AuditTrailConfig),
    ("ml_integration_config", MLIntegrationConfig),
    ("rlm_cognitive_config", RLMCognitiveConfig),
    ("checkpoint_memory_config", CheckpointMemoryConfig),
    ("cross_pollination_config", CrossPollinationConfig),
    ("km_bidirectional_config", KMBidirectionalConfig),
    ("translation_sub_config", TranslationSubConfig),
]

for _attr_name, _cls in _SUB_CONFIG_ATTRS:
    for _f in dataclass_fields(_cls):
        _SUB_CONFIG_GROUPS[_f.name] = (_attr_name, _cls)


# =============================================================================
# ArenaConfig Builder
# =============================================================================


class ArenaConfigBuilder:
    """Fluent builder for ArenaConfig.

    Allows constructing an ArenaConfig step-by-step using logical groupings::

        config = (
            ArenaConfig.builder()
            .with_hooks(enable_yaml_hooks=True)
            .with_tracking(enable_position_ledger=True)
            .with_knowledge(enable_knowledge_retrieval=True)
            .with_audit_trail(enable_receipt_generation=True)
            .with_ml(enable_ml_delegation=True)
            .build()
        )
    """

    def __init__(self) -> None:
        self._kwargs: dict[str, Any] = {}

    def _merge(self, kwargs: dict[str, Any]) -> "ArenaConfigBuilder":
        self._kwargs.update(kwargs)
        return self

    # -- Top-level fields --

    def with_identity(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set identification fields (loop_id, strict_loop_scoping)."""
        return self._merge(kwargs)

    def with_core(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set core subsystem fields (memory, event_emitter, spectator, etc.)."""
        return self._merge(kwargs)

    # -- Sub-config groups --

    def with_hooks(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set hook configuration fields."""
        return self._merge(kwargs)

    def with_tracking(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set tracking subsystem fields."""
        return self._merge(kwargs)

    def with_knowledge(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set Knowledge Mound integration fields."""
        return self._merge(kwargs)

    def with_memory_coordination(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set memory coordination fields."""
        return self._merge(kwargs)

    def with_performance_feedback(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set performance feedback loop fields."""
        return self._merge(kwargs)

    def with_audit_trail(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set audit trail fields (receipts, provenance, beads)."""
        return self._merge(kwargs)

    def with_ml(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set ML integration fields."""
        return self._merge(kwargs)

    def with_rlm(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set RLM cognitive load limiter fields."""
        return self._merge(kwargs)

    def with_checkpoint(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set checkpoint and memory state fields."""
        return self._merge(kwargs)

    def with_cross_pollination(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set Phase 9 cross-pollination bridge fields."""
        return self._merge(kwargs)

    def with_km_bidirectional(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set Phase 10 bidirectional KM fields."""
        return self._merge(kwargs)

    def with_translation(self, **kwargs: Any) -> "ArenaConfigBuilder":
        """Set multi-language translation fields."""
        return self._merge(kwargs)

    def build(self) -> "ArenaConfig":
        """Build the ArenaConfig from accumulated settings."""
        return ArenaConfig(**self._kwargs)


# =============================================================================
# ArenaConfig (main configuration class)
# =============================================================================


class ArenaConfig:
    """Configuration for Arena debate orchestration.

    Groups optional dependencies and settings that can be passed to Arena.
    This allows for cleaner initialization and easier testing.

    All fields from the sub-config dataclasses are accessible directly
    on this class for backward compatibility::

        config = ArenaConfig(enable_receipt_generation=True)
        assert config.enable_receipt_generation is True  # works

    Sub-config objects are also accessible for grouped usage::

        config.audit_trail_config.enable_receipt_generation  # also works

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

    Builder pattern::

        config = (
            ArenaConfig.builder()
            .with_identity(loop_id="debate-123")
            .with_knowledge(enable_knowledge_retrieval=True)
            .with_audit_trail(enable_receipt_generation=True)
            .build()
        )
    """

    # Keep __slots__ off so property delegation works with __dict__.
    # We use __init__ directly to support both flat kwargs and sub-configs.

    def __init__(
        self,
        # Identification
        loop_id: str = "",
        strict_loop_scoping: bool = False,
        # Core subsystems (typically injected)
        memory: Any | None = None,
        event_emitter: EventEmitterProtocol | None = None,
        spectator: SpectatorStream | None = None,
        debate_embeddings: DebateEmbeddingsProtocol | None = None,
        insight_store: InsightStoreProtocol | None = None,
        recorder: Any | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        evidence_collector: EvidenceCollectorProtocol | None = None,
        # Skills system integration
        skill_registry: Any | None = None,
        enable_skills: bool = False,
        # Propulsion engine (Gastown pattern)
        propulsion_engine: Any | None = None,
        enable_propulsion: bool = False,
        # Agent configuration
        agent_weights: Optional[dict[str, float]] = None,
        # Vertical personas
        vertical: str | None = None,
        vertical_persona_manager: Any | None = None,
        auto_detect_vertical: bool = True,
        # Performance telemetry
        performance_monitor: Any | None = None,
        enable_performance_monitor: bool = True,
        enable_telemetry: bool = False,
        # Agent selection
        agent_selector: Any | None = None,
        use_performance_selection: bool = True,
        # Airlock resilience
        use_airlock: bool = False,
        airlock_config: Any | None = None,
        # Prompt evolution
        prompt_evolver: PromptEvolverProtocol | None = None,
        enable_prompt_evolution: bool = False,
        # Billing/usage
        org_id: str = "",
        user_id: str = "",
        usage_tracker: Any | None = None,
        # Broadcast
        broadcast_pipeline: BroadcastPipelineProtocol | None = None,
        auto_broadcast: bool = False,
        broadcast_min_confidence: float = 0.8,
        broadcast_platforms: Optional[list[str]] = None,
        # Training data export
        training_exporter: Any | None = None,
        auto_export_training: bool = False,
        training_export_min_confidence: float = 0.75,
        training_export_path: str = "",
        # Genesis evolution
        population_manager: PopulationManagerProtocol | None = None,
        auto_evolve: bool = False,
        breeding_threshold: float = 0.8,
        # Fork/continuation
        initial_messages: Optional[list[Any]] = None,
        trending_topic: Any | None = None,
        pulse_manager: PulseManagerProtocol | None = None,
        auto_fetch_trending: bool = False,
        # Human-in-the-loop breakpoints
        breakpoint_manager: Any | None = None,
        # Post-debate workflow automation
        post_debate_workflow: Any | None = None,
        enable_post_debate_workflow: bool = False,
        post_debate_workflow_threshold: float = 0.7,
        # N+1 Query Detection
        enable_n1_detection: bool = False,
        n1_detection_mode: str = "warn",
        n1_detection_threshold: int = 5,
        # ---- Sub-config objects (optional, for grouped construction) ----
        hook_config: HookConfig | None = None,
        tracking_config: TrackingConfig | None = None,
        knowledge_config: KnowledgeMoundConfig | None = None,
        memory_coordination_config: MemoryCoordinationConfig | None = None,
        performance_feedback_config: PerformanceFeedbackConfig | None = None,
        audit_trail_config: AuditTrailConfig | None = None,
        ml_integration_config: MLIntegrationConfig | None = None,
        rlm_cognitive_config: RLMCognitiveConfig | None = None,
        checkpoint_memory_config: CheckpointMemoryConfig | None = None,
        cross_pollination_config: CrossPollinationConfig | None = None,
        km_bidirectional_config: KMBidirectionalConfig | None = None,
        translation_sub_config: TranslationSubConfig | None = None,
        # ---- Flat kwargs that belong to sub-configs (backward compat) ----
        **kwargs: Any,
    ) -> None:
        # -- Top-level fields (not in any sub-config) --
        self.loop_id = loop_id
        self.strict_loop_scoping = strict_loop_scoping
        self.memory = memory
        self.event_emitter = event_emitter
        self.spectator = spectator
        self.debate_embeddings = debate_embeddings
        self.insight_store = insight_store
        self.recorder = recorder
        self.circuit_breaker = circuit_breaker
        self.evidence_collector = evidence_collector
        self.skill_registry = skill_registry
        self.enable_skills = enable_skills
        self.propulsion_engine = propulsion_engine
        self.enable_propulsion = enable_propulsion
        self.agent_weights = agent_weights
        self.vertical = vertical
        self.vertical_persona_manager = vertical_persona_manager
        self.auto_detect_vertical = auto_detect_vertical
        self.performance_monitor = performance_monitor
        self.enable_performance_monitor = enable_performance_monitor
        self.enable_telemetry = enable_telemetry
        self.agent_selector = agent_selector
        self.use_performance_selection = use_performance_selection
        self.use_airlock = use_airlock
        self.airlock_config = airlock_config
        self.prompt_evolver = prompt_evolver
        self.enable_prompt_evolution = enable_prompt_evolution
        self.org_id = org_id
        self.user_id = user_id
        self.usage_tracker = usage_tracker
        self.broadcast_pipeline = broadcast_pipeline
        self.auto_broadcast = auto_broadcast
        self.broadcast_min_confidence = broadcast_min_confidence
        self.broadcast_platforms = broadcast_platforms
        self.training_exporter = training_exporter
        self.auto_export_training = auto_export_training
        self.training_export_min_confidence = training_export_min_confidence
        self.training_export_path = training_export_path
        self.population_manager = population_manager
        self.auto_evolve = auto_evolve
        self.breeding_threshold = breeding_threshold
        self.initial_messages = initial_messages
        self.trending_topic = trending_topic
        self.pulse_manager = pulse_manager
        self.auto_fetch_trending = auto_fetch_trending
        self.breakpoint_manager = breakpoint_manager
        self.post_debate_workflow = post_debate_workflow
        self.enable_post_debate_workflow = enable_post_debate_workflow
        self.post_debate_workflow_threshold = post_debate_workflow_threshold
        self.enable_n1_detection = enable_n1_detection
        self.n1_detection_mode = n1_detection_mode
        self.n1_detection_threshold = n1_detection_threshold

        # -- Build sub-configs from flat kwargs + explicit sub-config objects --
        # For each sub-config group, collect any flat kwargs that belong to it,
        # then merge with an explicit sub-config object if provided.
        self.hook_config = self._build_sub_config(HookConfig, hook_config, kwargs)
        self.tracking_config = self._build_sub_config(TrackingConfig, tracking_config, kwargs)
        self.knowledge_config = self._build_sub_config(
            KnowledgeMoundConfig, knowledge_config, kwargs
        )
        self.memory_coordination_config = self._build_sub_config(
            MemoryCoordinationConfig, memory_coordination_config, kwargs
        )
        self.performance_feedback_config = self._build_sub_config(
            PerformanceFeedbackConfig, performance_feedback_config, kwargs
        )
        self.audit_trail_config = self._build_sub_config(
            AuditTrailConfig, audit_trail_config, kwargs
        )
        self.ml_integration_config = self._build_sub_config(
            MLIntegrationConfig, ml_integration_config, kwargs
        )
        self.rlm_cognitive_config = self._build_sub_config(
            RLMCognitiveConfig, rlm_cognitive_config, kwargs
        )
        self.checkpoint_memory_config = self._build_sub_config(
            CheckpointMemoryConfig, checkpoint_memory_config, kwargs
        )
        self.cross_pollination_config = self._build_sub_config(
            CrossPollinationConfig, cross_pollination_config, kwargs
        )
        self.km_bidirectional_config = self._build_sub_config(
            KMBidirectionalConfig, km_bidirectional_config, kwargs
        )
        self.translation_sub_config = self._build_sub_config(
            TranslationSubConfig, translation_sub_config, kwargs
        )

        # Any remaining kwargs are unknown fields
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"ArenaConfig received unknown keyword arguments: {unknown}")

        # Post-init defaults
        if self.broadcast_platforms is None:
            self.broadcast_platforms = ["rss"]

    @staticmethod
    def _build_sub_config(
        cls: type,
        explicit: Any | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """Build a sub-config from flat kwargs, optionally overlaying on an explicit instance.

        If an explicit sub-config object is provided, flat kwargs override its values.
        If no explicit object is provided, a new one is created from defaults + kwargs.
        """
        field_names = {f.name for f in dataclass_fields(cls)}
        overrides = {}
        for name in list(kwargs):
            if name in field_names:
                overrides[name] = kwargs.pop(name)

        if explicit is not None:
            # Start from the explicit object's values, then apply overrides
            init_kwargs = {}
            for f in dataclass_fields(cls):
                if f.name in overrides:
                    init_kwargs[f.name] = overrides[f.name]
                else:
                    init_kwargs[f.name] = getattr(explicit, f.name)
            return cls(**init_kwargs)
        else:
            # Create from defaults + overrides
            return cls(**overrides)

    # =========================================================================
    # Backward-compatible attribute access via __getattr__
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to sub-configs for backward compatibility.

        This allows ``config.enable_receipt_generation`` to transparently
        access ``config.audit_trail_config.enable_receipt_generation``.
        """
        # Look up which sub-config owns this field
        mapping = _SUB_CONFIG_GROUPS.get(name)
        if mapping is not None:
            attr_name, _ = mapping
            sub = object.__getattribute__(self, attr_name)
            return getattr(sub, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to sub-configs for backward compatibility.

        This allows ``config.enable_receipt_generation = True`` to transparently
        set ``config.audit_trail_config.enable_receipt_generation = True``.
        """
        # During __init__, allow setting all attributes directly
        # After __init__, delegate sub-config fields
        mapping = _SUB_CONFIG_GROUPS.get(name)
        if mapping is not None:
            attr_name, _ = mapping
            # Check if the sub-config attribute exists yet (it may not during __init__)
            try:
                sub = object.__getattribute__(self, attr_name)
                setattr(sub, name, value)
                return
            except AttributeError:
                # Sub-config not yet initialized, fall through to direct set
                pass
        object.__setattr__(self, name, value)

    # =========================================================================
    # Equality and repr (dataclass-like behavior)
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArenaConfig):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        parts = []
        for key, value in self.__dict__.items():
            # Skip sub-configs that are at default values to keep repr manageable
            if key.endswith("_config") and key != "airlock_config":
                mapping = {a: c for a, c in _SUB_CONFIG_ATTRS}
                cls = mapping.get(key)
                if cls is not None:
                    default = cls()
                    if value == default:
                        continue
            parts.append(f"{key}={value!r}")
        return f"ArenaConfig({', '.join(parts)})"

    # =========================================================================
    # Builder factory
    # =========================================================================

    @classmethod
    def builder(cls) -> ArenaConfigBuilder:
        """Create a new ArenaConfigBuilder for fluent construction."""
        return ArenaConfigBuilder()

    # =========================================================================
    # to_arena_kwargs (preserved from original)
    # =========================================================================

    def to_arena_kwargs(self) -> dict[str, Any]:
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
            "enable_knowledge_extraction": self.enable_knowledge_extraction,
            "extraction_min_confidence": self.extraction_min_confidence,
            # Auto-revalidation
            "enable_auto_revalidation": self.enable_auto_revalidation,
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
            "skill_registry": self.skill_registry,
            "enable_skills": self.enable_skills,
            "propulsion_engine": self.propulsion_engine,
            "enable_propulsion": self.enable_propulsion,
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


# ===========================================================================
# Primary Config Groups (for Arena constructor refactoring)
# ===========================================================================
# These are the primary configuration groups for cleaner Arena initialization.
# They group related parameters into cohesive units per the refactoring spec:
# - DebateConfig (rounds, consensus threshold, protocol settings)
# - AgentConfig (agent list, fallback agents, circuit breaker settings)
# - MemoryConfig (memory systems, critique store, wisdom store)
# - StreamingConfig (WebSocket settings, event hooks)
# - ObservabilityConfig (telemetry, performance monitor, immune system)


@dataclass
class DebateConfig:
    """Configuration for debate protocol settings.

    Groups parameters related to debate rounds, consensus detection,
    and protocol-level behavior.

    Example::

        debate_config = DebateConfig(
            rounds=5,
            consensus_threshold=0.7,
            enable_adaptive_rounds=True,
        )
        arena = Arena(env, agents, debate_config=debate_config)
    """

    # Protocol settings (passed to DebateProtocol or override defaults)
    rounds: int = DEFAULT_ROUNDS  # Number of debate rounds
    consensus_threshold: float = 0.7  # Threshold for consensus detection
    convergence_detection: bool = True  # Enable semantic convergence detection
    convergence_threshold: float = 0.85  # Similarity threshold for convergence
    divergence_threshold: float = 0.3  # Threshold for divergence detection
    timeout_seconds: int = 0  # Overall debate timeout (0 = no timeout)
    judge_selection: str = "elo_ranked"  # Judge selection mode

    # Adaptive debate settings
    enable_adaptive_rounds: bool = False  # Use memory-based strategy for rounds
    debate_strategy: Any | None = None  # Pre-configured DebateStrategy instance

    # Early termination settings
    enable_judge_termination: bool = False  # Allow judge to terminate early
    enable_early_stopping: bool = False  # Allow agents to request early stop

    # Hierarchy settings (Gastown pattern)
    enable_agent_hierarchy: bool = True  # Assign orchestrator/monitor/worker roles
    hierarchy_config: Any | None = None  # Optional HierarchyConfig

    def apply_to_protocol(self, protocol: "DebateProtocol") -> "DebateProtocol":
        """Apply config values to a DebateProtocol instance."""
        protocol.rounds = self.rounds
        protocol.consensus_threshold = self.consensus_threshold
        protocol.convergence_detection = self.convergence_detection
        protocol.convergence_threshold = self.convergence_threshold
        protocol.divergence_threshold = self.divergence_threshold
        protocol.timeout_seconds = self.timeout_seconds
        protocol.judge_selection = self.judge_selection
        return protocol


@dataclass
class AgentConfig:
    """Configuration for agent management and selection.

    Groups parameters related to agent pool management, performance-based
    selection, and resilience settings.

    Example::

        agent_config = AgentConfig(
            use_performance_selection=True,
            use_airlock=True,
            agent_weights={"claude": 1.2, "gpt4": 1.0},
        )
        arena = Arena(env, agents, agent_config=agent_config)
    """

    # Agent weights and selection
    agent_weights: Optional[dict[str, float]] = None  # Reliability weights
    agent_selector: Any | None = None  # AgentSelector for performance-based selection
    use_performance_selection: bool = True  # Enable ELO/calibration-based selection

    # Resilience settings
    circuit_breaker: CircuitBreaker | None = None  # Circuit breaker for failure handling
    use_airlock: bool = False  # Wrap agents with AirlockProxy for timeout protection
    airlock_config: Any | None = None  # AirlockConfig customization

    # Tracking subsystems
    position_tracker: PositionTrackerProtocol | None = None
    position_ledger: PositionLedgerProtocol | None = None
    enable_position_ledger: bool = False  # Auto-create PositionLedger
    elo_system: EloSystemProtocol | None = None
    calibration_tracker: CalibrationTrackerProtocol | None = None
    relationship_tracker: RelationshipTrackerProtocol | None = None

    # Personas
    persona_manager: PersonaManagerProtocol | None = None
    vertical: str | None = None  # Industry vertical
    vertical_persona_manager: Any | None = None
    auto_detect_vertical: bool = True

    # Agent Fabric (high-scale orchestration)
    fabric: Any | None = None  # AgentFabric instance
    fabric_config: Any | None = None  # FabricDebateConfig


@dataclass
class StreamingConfig:
    """Configuration for WebSocket streaming and event handling.

    Groups parameters related to real-time event emission, spectator
    streams, and event hooks.

    Example::

        streaming_config = StreamingConfig(
            spectator=spectator_stream,
            event_emitter=emitter,
            loop_id="debate-123",
        )
        arena = Arena(env, agents, streaming_config=streaming_config)
    """

    # Event hooks and emitters
    event_hooks: Optional[dict[str, Any]] = None  # Hooks for streaming events
    hook_manager: Any | None = None  # HookManager for extended lifecycle
    event_emitter: EventEmitterProtocol | None = None  # User event subscription

    # Spectator streaming
    spectator: SpectatorStream | None = None  # Real-time event stream

    # Recording
    recorder: Any | None = None  # ReplayRecorder for debate recording

    # Loop scoping
    loop_id: str = ""  # Loop ID for multi-loop scoping
    strict_loop_scoping: bool = False  # Drop events without loop_id

    # Skills integration
    skill_registry: Any | None = None  # SkillRegistry for extensible capabilities
    enable_skills: bool = False  # Enable skills during evidence collection

    # Propulsion (push-based work assignment)
    propulsion_engine: Any | None = None  # PropulsionEngine
    enable_propulsion: bool = False  # Enable propulsion events


@dataclass
class ObservabilityConfig:
    """Configuration for telemetry, monitoring, and health systems.

    Groups parameters related to performance monitoring, Prometheus metrics,
    and billing integration.

    Example::

        observability_config = ObservabilityConfig(
            enable_telemetry=True,
            enable_performance_monitor=True,
            org_id="org-123",
        )
        arena = Arena(env, agents, observability_config=observability_config)
    """

    # Performance monitoring
    performance_monitor: Any | None = None  # AgentPerformanceMonitor
    enable_performance_monitor: bool = True  # Auto-create monitor
    enable_telemetry: bool = False  # Enable Prometheus metrics

    # Prompt evolution
    prompt_evolver: PromptEvolverProtocol | None = None  # Pattern extraction
    enable_prompt_evolution: bool = False  # Auto-create evolver

    # Breakpoints (human-in-the-loop)
    breakpoint_manager: Any | None = None  # BreakpointManager

    # Trending topics / Pulse integration
    trending_topic: Any | None = None  # TrendingTopic to seed context
    pulse_manager: PulseManagerProtocol | None = None  # Auto-fetch trending
    auto_fetch_trending: bool = False  # Auto-fetch if none provided

    # Evolution / breeding
    population_manager: PopulationManagerProtocol | None = None  # Genome evolution
    auto_evolve: bool = False  # Trigger evolution after quality debates
    breeding_threshold: float = 0.8  # Min confidence for evolution

    # Evidence collection
    evidence_collector: EvidenceCollectorProtocol | None = None

    # Billing / usage tracking
    org_id: str = ""  # Organization ID for multi-tenancy
    user_id: str = ""  # User ID for attribution
    usage_tracker: Any | None = None  # UsageTracker

    # Broadcast pipeline
    broadcast_pipeline: BroadcastPipelineProtocol | None = None
    auto_broadcast: bool = False  # Auto-trigger broadcast
    broadcast_min_confidence: float = 0.8  # Min confidence for broadcast

    # Training export
    training_exporter: Any | None = None  # DebateTrainingExporter
    auto_export_training: bool = False  # Auto-export training data
    training_export_min_confidence: float = 0.75  # Min confidence to export

    # ML Integration
    enable_ml_delegation: bool = False  # ML-based agent selection
    ml_delegation_strategy: Any | None = None  # MLDelegationStrategy
    ml_delegation_weight: float = 0.3  # Weight for ML vs ELO
    enable_quality_gates: bool = False  # Filter low-quality responses
    quality_gate_threshold: float = 0.6  # Min quality score
    enable_consensus_estimation: bool = False  # Early termination estimation
    consensus_early_termination_threshold: float = 0.85  # Probability threshold

    # Post-debate workflow
    post_debate_workflow: Any | None = None  # Workflow DAG
    enable_post_debate_workflow: bool = False  # Auto-trigger workflow
    post_debate_workflow_threshold: float = 0.7  # Min confidence for workflow

    # Fork/continuation
    initial_messages: Optional[list[Any]] = None  # Initial conversation history


# Primary config classes tuple for iteration
PRIMARY_CONFIG_CLASSES = (
    DebateConfig,
    AgentConfig,
    StreamingConfig,
    ObservabilityConfig,
)


# ===========================================================================
# Decomposed Config Groups (Legacy)
# ===========================================================================
# These provide focused configuration for specific Arena subsystems.
# They can be passed individually to Arena or composed into an ArenaConfig.
# Note: Some overlap with primary configs above - both are supported.


@dataclass
class MemoryConfig:
    """Memory subsystem configuration.

    Groups parameters related to critique storage, continuum memory,
    consensus memory, and knowledge mound integration.

    Example::

        memory_config = MemoryConfig(
            enable_knowledge_retrieval=True,
            enable_cross_debate_memory=True,
            auto_create_knowledge_mound=True,
        )
        arena = Arena(env, agents, memory_config=memory_config)
    """

    # Core memory stores
    memory: Any | None = None  # CritiqueStore instance
    continuum_memory: ContinuumMemoryProtocol | None = None  # Cross-debate learning
    consensus_memory: ConsensusMemoryProtocol | None = None  # Historical outcomes
    debate_embeddings: DebateEmbeddingsProtocol | None = None  # Historical context
    insight_store: InsightStoreProtocol | None = None  # Debate learnings
    dissent_retriever: DissentRetrieverProtocol | None = None  # Historical minority views
    flip_detector: FlipDetectorProtocol | None = None  # Position reversal detection
    moment_detector: MomentDetectorProtocol | None = None  # Significant moments
    tier_analytics_tracker: Any | None = None  # Memory ROI tracking

    # Cross-debate memory
    cross_debate_memory: Any | None = None  # Institutional knowledge
    enable_cross_debate_memory: bool = True  # Inject institutional knowledge

    # Knowledge Mound integration
    knowledge_mound: Any | None = None  # KnowledgeMound instance
    auto_create_knowledge_mound: bool = True  # Auto-create if not provided
    enable_knowledge_retrieval: bool = True  # Query mound before debates
    enable_knowledge_ingestion: bool = True  # Store consensus outcomes
    enable_knowledge_extraction: bool = False  # Extract structured claims
    extraction_min_confidence: float = 0.3  # Min confidence for extraction
    enable_belief_guidance: bool = False  # Inject historical cruxes

    # Revalidation settings
    enable_auto_revalidation: bool = False  # Auto-trigger for stale knowledge
    revalidation_staleness_threshold: float = 0.8  # Staleness threshold
    revalidation_check_interval_seconds: int = 3600  # Check interval
    revalidation_scheduler: Any | None = None  # RevalidationScheduler instance

    # RLM cognitive load limiter
    use_rlm_limiter: bool = True  # Use RLM for context compression
    rlm_limiter: Any | None = None  # Pre-configured limiter
    rlm_compression_threshold: int = 3000  # Chars to trigger compression
    rlm_max_recent_messages: int = 5  # Recent messages at full detail
    rlm_summary_level: str = "SUMMARY"  # Abstraction level
    rlm_compression_round_threshold: int = 3  # Auto-compression after N rounds

    # Checkpointing
    checkpoint_manager: Any | None = None  # CheckpointManager for resume
    enable_checkpointing: bool = True  # Auto-create CheckpointManager


@dataclass
class KnowledgeConfig:
    """Knowledge Mound configuration."""

    knowledge_mound: Any | None = None
    auto_create_knowledge_mound: bool = True
    enable_knowledge_retrieval: bool = True
    enable_knowledge_ingestion: bool = True
    enable_knowledge_extraction: bool = False
    extraction_min_confidence: float = 0.3
    enable_auto_revalidation: bool = False
    revalidation_staleness_threshold: float = 0.7
    revalidation_check_interval_seconds: int = 3600
    revalidation_scheduler: Any | None = None
    enable_belief_guidance: bool = False


@dataclass
class MLConfig:
    """Machine learning integration configuration."""

    enable_ml_delegation: bool = False
    ml_delegation_strategy: Any | None = None
    ml_delegation_weight: float = 0.3
    enable_quality_gates: bool = False
    quality_gate_threshold: float = 0.6
    enable_consensus_estimation: bool = False
    consensus_early_termination_threshold: float = 0.85


@dataclass
class RLMConfig:
    """RLM cognitive load limiter configuration."""

    use_rlm_limiter: bool = True
    rlm_limiter: Any | None = None
    rlm_compression_threshold: int = 3000
    rlm_max_recent_messages: int = 5
    rlm_summary_level: str = "SUMMARY"
    rlm_compression_round_threshold: int = 3


@dataclass
class TelemetryConfig:
    """Telemetry and performance monitoring configuration."""

    performance_monitor: Any | None = None
    enable_performance_monitor: bool = True
    enable_telemetry: bool = False


@dataclass
class PersonaConfig:
    """Persona and vertical configuration."""

    persona_manager: PersonaManagerProtocol | None = None
    vertical: str | None = None
    auto_detect_vertical: bool = True
    vertical_persona_manager: Any | None = None


@dataclass
class ResilienceConfig:
    """Resilience and fault tolerance configuration."""

    circuit_breaker: CircuitBreaker | None = None
    use_airlock: bool = False
    airlock_config: Any | None = None


@dataclass
class EvolutionConfig:
    """Evolution and prompt optimization configuration."""

    population_manager: PopulationManagerProtocol | None = None
    auto_evolve: bool = False
    breeding_threshold: float = 0.8
    prompt_evolver: PromptEvolverProtocol | None = None
    enable_prompt_evolution: bool = False


@dataclass
class BillingConfig:
    """Billing and usage tracking configuration."""

    org_id: str = ""
    user_id: str = ""
    usage_tracker: Any | None = None


@dataclass
class BroadcastConfig:
    """Broadcast and training export configuration."""

    broadcast_pipeline: BroadcastPipelineProtocol | None = None
    auto_broadcast: bool = False
    broadcast_min_confidence: float = 0.8
    broadcast_platforms: Optional[list[str]] = None
    training_exporter: Any | None = None
    auto_export_training: bool = False
    training_export_min_confidence: float = 0.75


@dataclass
class TranslationConfig:
    """Multi-language translation configuration."""

    translation_service: Any | None = None  # Pre-configured TranslationService
    multilingual_manager: Any | None = None  # Pre-configured MultilingualDebateManager
    enable_translation: bool = False  # Enable multi-language debate support
    default_language: str = "en"  # Default language code (ISO 639-1)
    target_languages: Optional[list[str]] = None  # Languages to translate conclusions to
    auto_detect_language: bool = True  # Auto-detect source language of messages
    translate_conclusions: bool = True  # Translate final conclusions to target languages
    translation_cache_ttl_seconds: int = 3600  # Translation cache TTL (1 hour default)
    translation_cache_max_entries: int = 10000  # Max entries in translation cache


# Legacy config classes tuple (preserved for compatibility)
LEGACY_CONFIG_CLASSES = (
    KnowledgeConfig,
    MLConfig,
    RLMConfig,
    TelemetryConfig,
    PersonaConfig,
    ResilienceConfig,
    EvolutionConfig,
    BillingConfig,
    BroadcastConfig,
    TranslationConfig,
)

# All config classes (primary + legacy + MemoryConfig which is in both)
ALL_CONFIG_CLASSES = PRIMARY_CONFIG_CLASSES + (MemoryConfig,) + LEGACY_CONFIG_CLASSES

# Sub-config classes (new pattern) for direct import
SUB_CONFIG_CLASSES = (
    HookConfig,
    TrackingConfig,
    KnowledgeMoundConfig,
    MemoryCoordinationConfig,
    PerformanceFeedbackConfig,
    AuditTrailConfig,
    MLIntegrationConfig,
    RLMCognitiveConfig,
    CheckpointMemoryConfig,
    CrossPollinationConfig,
    KMBidirectionalConfig,
    TranslationSubConfig,
)


__all__ = [
    "ArenaConfig",
    "ArenaConfigBuilder",
    # Sub-config classes (new strategy/builder pattern)
    "HookConfig",
    "TrackingConfig",
    "KnowledgeMoundConfig",
    "MemoryCoordinationConfig",
    "PerformanceFeedbackConfig",
    "AuditTrailConfig",
    "MLIntegrationConfig",
    "RLMCognitiveConfig",
    "CheckpointMemoryConfig",
    "CrossPollinationConfig",
    "KMBidirectionalConfig",
    "TranslationSubConfig",
    "SUB_CONFIG_CLASSES",
    # Primary config classes (for Arena constructor refactoring)
    "DebateConfig",
    "AgentConfig",
    "MemoryConfig",
    "StreamingConfig",
    "ObservabilityConfig",
    # Legacy config classes
    "KnowledgeConfig",
    "MLConfig",
    "RLMConfig",
    "TelemetryConfig",
    "PersonaConfig",
    "ResilienceConfig",
    "EvolutionConfig",
    "BillingConfig",
    "BroadcastConfig",
    "TranslationConfig",
    # Config class collections
    "PRIMARY_CONFIG_CLASSES",
    "LEGACY_CONFIG_CLASSES",
    "ALL_CONFIG_CLASSES",
]
