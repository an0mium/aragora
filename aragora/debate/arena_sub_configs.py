"""Sub-configuration dataclasses for ArenaConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aragora.type_protocols import (
    CalibrationTrackerProtocol,
    ConsensusMemoryProtocol,
    ContinuumMemoryProtocol,
    DissentRetrieverProtocol,
    EloSystemProtocol,
    FlipDetectorProtocol,
    MomentDetectorProtocol,
    PersonaManagerProtocol,
    PositionLedgerProtocol,
    PositionTrackerProtocol,
    RelationshipTrackerProtocol,
)


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

    event_hooks: dict[str, Any] | None = None
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

    # Trickster Domain-Specific Thresholds (Pillar 4: Multi-agent robustness)
    trickster_domain_configs: dict[str, Any] | None = None  # Domain -> TricksterConfig mapping


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

    # Compliance Artifact Generation (EU AI Act)
    enable_compliance_artifacts: bool = False  # Auto-generate compliance artifacts
    compliance_frameworks: list[str] | None = None  # Frameworks to generate for


@dataclass
class MLIntegrationConfig:
    """ML integration configuration.

    Groups parameters for ML-based agent delegation, quality gates,
    and consensus estimation for early termination.

    All three ML features (delegation, quality gates, consensus estimation)
    are enabled by default as stable features.

    Example::

        ml_cfg = MLIntegrationConfig(
            enable_ml_delegation=True,
            ml_delegation_weight=0.5,
            enable_quality_gates=True,
        )
    """

    enable_ml_delegation: bool = True  # Use ML-based agent selection
    ml_delegation_strategy: Any | None = None  # Custom MLDelegationStrategy
    ml_delegation_weight: float = 0.3  # Weight for ML scoring vs ELO (0.0-1.0)
    enable_quality_gates: bool = True  # Filter low-quality responses via QualityGate
    quality_gate_threshold: float = 0.6  # Minimum quality score (0.0-1.0)
    enable_consensus_estimation: bool = True  # Use ConsensusEstimator for early termination
    consensus_early_termination_threshold: float = 0.85  # Probability threshold for early stop
    enable_stability_detection: bool = False  # Use adaptive stability detection
    stability_threshold: float = 0.85  # Stability threshold for early stop
    stability_min_rounds: int = 2  # Minimum rounds before stability stop
    stability_agreement_threshold: float = 0.75  # Agreement threshold per round
    stability_conflict_confidence: float = 0.7  # ML confidence to override stability


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
    target_languages: list[str] | None = None  # Languages to translate conclusions to
    auto_detect_language: bool = True  # Auto-detect source language of messages
    translate_conclusions: bool = True  # Translate final conclusions to target languages
    translation_cache_ttl_seconds: int = 3600  # Translation cache TTL (1 hour default)
    translation_cache_max_entries: int = 10000  # Max entries in translation cache


@dataclass
class SupermemorySubConfig:
    """Supermemory external memory integration configuration.

    Groups parameters for Supermemory integration, enabling cross-session
    learning and context injection from external memory.

    Supermemory provides persistent memory across projects:
    - Context injection on debate start (reverse flow)
    - Outcome persistence after debate (forward flow)
    - Semantic search across historical memories

    Example::

        sm_cfg = SupermemorySubConfig(
            enable_supermemory=True,
            supermemory_inject_on_start=True,
            supermemory_sync_on_conclusion=True,
        )
    """

    # Master switch (opt-in, disabled by default)
    enable_supermemory: bool = False

    # KM adapter toggle (auto-enable adapter in bidirectional coordinator)
    supermemory_enable_km_adapter: bool = False

    # Pre-configured adapter instance (optional)
    supermemory_adapter: Any | None = None

    # Context injection settings (reverse flow: Supermemory -> Aragora)
    supermemory_inject_on_start: bool = True  # Inject context at debate start
    supermemory_max_context_items: int = 10  # Max memories to inject
    supermemory_context_container_tag: str | None = None  # Container to query

    # Outcome sync settings (forward flow: Aragora -> Supermemory)
    supermemory_sync_on_conclusion: bool = True  # Sync outcome after debate
    supermemory_min_confidence_for_sync: float = 0.7  # Min confidence to sync
    supermemory_outcome_container_tag: str | None = None  # Container for outcomes

    # Privacy settings
    supermemory_enable_privacy_filter: bool = True  # Filter sensitive data before sync

    # Resilience settings
    supermemory_enable_resilience: bool = True  # Use circuit breaker protection


@dataclass
class BudgetSubConfig:
    """Per-debate budget configuration.

    Groups parameters for cost control during debate execution.
    When budget_limit_usd is set, the debate will enforce cost limits
    and can trigger early termination or model switching.

    Example::

        budget_cfg = BudgetSubConfig(
            budget_limit_usd=2.00,
            budget_alert_threshold=0.75,
            budget_hard_stop=True,
        )
    """

    # Master budget cap for the debate (None = unlimited)
    budget_limit_usd: float | None = None

    # Fraction of budget at which to emit a warning event (0.0-1.0)
    budget_alert_threshold: float = 0.75

    # If True, hard-stop the debate when budget is exceeded.
    # If False, allow the current round to finish.
    budget_hard_stop: bool = False

    # If True, switch to cheaper models when approaching the limit.
    budget_downgrade_models: bool = False

    # Per-round cost cap (None = no per-round limit)
    budget_per_round_usd: float | None = None


@dataclass
class PowerSamplingConfig:
    """Power sampling configuration for inference-time reasoning.

    Enables best-of-n sampling with power-law weighted selection during
    proposal generation. Instead of generating a single response, agents
    generate multiple samples and select the best one based on quality
    scoring and diversity.

    Based on research showing that inference-time compute can significantly
    improve reasoning quality even without additional training.

    Example::

        power_cfg = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=8,
            alpha=2.0,
        )
    """

    # Master switch for power sampling
    enable_power_sampling: bool = False

    # Number of samples to generate per proposal (higher = better quality, more cost)
    n_samples: int = 8

    # Power law exponent (higher = more concentrated on top samples)
    alpha: float = 2.0

    # Number of diverse samples to consider for final selection
    k_diverse: int = 3

    # Temperature for generation (higher = more diversity between samples)
    sampling_temperature: float = 1.0

    # Minimum quality score to accept a sample (0.0-1.0)
    min_quality_threshold: float = 0.3

    # Whether to apply power sampling to critique phase too
    enable_for_critiques: bool = False

    # Custom scorer function path (e.g., "mymodule.custom_scorer")
    # If None, uses the default quality scorer
    custom_scorer: str | None = None

    # Per-sample timeout in seconds (lower timeout for faster sampling)
    sample_timeout: float = 30.0


@dataclass
class AutoExecutionConfig:
    """Auto-execution of debate results via the Decision Pipeline.

    When enabled, the Arena will automatically generate a DecisionPlan from
    the debate result and optionally execute it through the PlanExecutor.

    The approval mode controls whether the plan is auto-executed or held
    for human review. Risk-based approval (default) will auto-execute only
    plans whose highest risk level is at or below ``auto_max_risk``.

    Example::

        exec_cfg = AutoExecutionConfig(
            enable_auto_execution=True,
            auto_execution_mode="workflow",
            auto_approval_mode="risk_based",
            auto_max_risk="low",
        )
    """

    # Master switch (opt-in, disabled by default)
    enable_auto_execution: bool = False

    # Execution mode passed to PlanExecutor
    # One of: "workflow", "hybrid", "fabric"
    auto_execution_mode: str = "workflow"

    # Maps to ApprovalMode on DecisionPlanFactory.from_debate_result
    # One of: "always", "risk_based", "confidence_based", "never"
    auto_approval_mode: str = "risk_based"

    # Maximum risk level for auto-execution without human approval
    # One of: "low", "medium", "high", "critical"
    auto_max_risk: str = "low"


@dataclass
class UnifiedMemorySubConfig:
    """Unified Memory Gateway configuration.

    Groups parameters for the cross-system memory gateway that provides
    fan-out queries, deduplication, and ranking across all memory systems
    (ContinuumMemory, Knowledge Mound, Supermemory, claude-mem).

    When enabled, the gateway provides a single query/store API with
    Titans/MIRAS-inspired retention decisions via the RetentionGate.

    Example::

        um_cfg = UnifiedMemorySubConfig(
            enable_unified_memory=True,
            enable_retention_gate=True,
        )
    """

    # Master switch (opt-in, disabled by default)
    enable_unified_memory: bool = False

    # Enable Titans-inspired surprise-driven retention decisions
    enable_retention_gate: bool = False

    # Query timeout for individual memory sources (seconds)
    query_timeout_seconds: float = 15.0

    # Dedup similarity threshold (0-1, higher = stricter matching)
    dedup_threshold: float = 0.95

    # Default sources to query (None = all available)
    default_sources: list[str] | None = None

    # Whether to run source queries in parallel
    parallel_queries: bool = True


__all__ = [
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
    "SupermemorySubConfig",
    "BudgetSubConfig",
    "PowerSamplingConfig",
    "AutoExecutionConfig",
    "UnifiedMemorySubConfig",
]
