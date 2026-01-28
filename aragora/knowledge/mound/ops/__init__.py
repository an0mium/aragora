"""
Knowledge Mound Operations mixins.

This module provides operational mixins for the KnowledgeMound facade:
- StalenessOperationsMixin: Staleness detection and revalidation
- CultureOperationsMixin: Culture accumulation and management
- SyncOperationsMixin: Cross-system synchronization
- GlobalKnowledgeMixin: Global/public knowledge operations
- KnowledgeSharingMixin: Cross-workspace knowledge sharing
- KnowledgeFederationMixin: Multi-region knowledge synchronization
- DedupOperationsMixin: Similarity-based deduplication
- PruningOperationsMixin: Automatic and manual pruning
- AutoCurationMixin: Intelligent automated knowledge maintenance (Phase 4)

Phase A2 Operations:
- ContradictionOperationsMixin: Contradiction detection and resolution
- ConfidenceDecayMixin: Dynamic confidence adjustment over time
- GovernanceMixin: RBAC and audit trails
- AnalyticsMixin: Coverage, usage, and quality analytics
"""

from aragora.knowledge.mound.ops.staleness import StalenessOperationsMixin
from aragora.knowledge.mound.ops.culture import CultureOperationsMixin
from aragora.knowledge.mound.ops.sync import SyncOperationsMixin
from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin, SYSTEM_WORKSPACE_ID
from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
from aragora.knowledge.mound.ops.federation import (
    KnowledgeFederationMixin,
    FederationMode,
    SyncScope,
    FederatedRegion,
    SyncResult,
)
from aragora.knowledge.mound.ops.dedup import (
    DedupOperationsMixin,
    DuplicateCluster,
    DuplicateMatch,
    DedupReport,
    MergeResult,
)
from aragora.knowledge.mound.ops.pruning import (
    PruningOperationsMixin,
    PruningPolicy,
    PrunableItem,
    PruneResult,
    PruneHistory,
    PruningAction,
)
from aragora.knowledge.mound.ops.auto_curation import (
    AutoCurationMixin,
    CurationPolicy,
    CurationCandidate,
    CurationResult,
    CurationHistory,
    CurationAction,
    QualityScore,
    TierLevel,
)

# Phase A2 Operations
from aragora.knowledge.mound.ops.contradiction import (
    ContradictionOperationsMixin,
    ContradictionDetector,
    Contradiction,
    ContradictionReport,
    ContradictionConfig,
    ContradictionType,
    ResolutionStrategy,
    get_contradiction_detector,
)
from aragora.knowledge.mound.ops.confidence_decay import (
    ConfidenceDecayMixin,
    ConfidenceDecayManager,
    ConfidenceAdjustment,
    DecayReport,
    DecayConfig,
    DecayModel,
    ConfidenceEvent,
    get_decay_manager,
)
from aragora.knowledge.mound.ops.governance import (
    GovernanceMixin,
    RBACManager,
    AuditTrail,
    Role,
    RoleAssignment,
    Permission,
    BuiltinRole,
    BUILTIN_ROLES,
    AuditAction,
    AuditEntry,
    get_rbac_manager,
    get_audit_trail,
)
from aragora.knowledge.mound.ops.analytics import (
    AnalyticsMixin,
    KnowledgeAnalytics,
    DomainCoverage,
    CoverageReport,
    UsageEvent,
    UsageEventType,
    ItemUsageStats,
    UsageReport,
    QualitySnapshot,
    QualityTrend,
    GrowthMetrics,
    get_knowledge_analytics,
)
from aragora.knowledge.mound.ops.extraction import (
    ExtractionMixin,
    DebateKnowledgeExtractor,
    ExtractedClaim,
    ExtractedRelationship,
    ExtractionResult,
    ExtractionConfig,
    ExtractionType,
    ConfidenceSource,
    get_debate_extractor,
)

# Phase A3 Operations
from aragora.knowledge.mound.ops.fusion import (
    FusionStrategy,
    ConflictResolution,
    AdapterValidation,
    FusedValidation,
    FusionConfig,
    FusionCoordinator,
    get_fusion_coordinator,
)
from aragora.knowledge.mound.ops.calibration_fusion import (
    CalibrationFusionStrategy,
    AgentPrediction,
    CalibrationConsensus,
    CalibrationFusionConfig,
    CalibrationFusionEngine,
    get_calibration_fusion_engine,
)
from aragora.knowledge.mound.ops.multi_party_validation import (
    ValidationVoteType,
    ValidationConsensusStrategy,
    ValidationState,
    ValidationVote,
    ValidationRequest,
    ValidationResult,
    EscalationResult,
    ValidatorConfig,
    MultiPartyValidator,
    get_multi_party_validator,
)
from aragora.knowledge.mound.ops.quality_signals import (
    QualityDimension,
    OverconfidenceLevel,
    QualityTier,
    CalibrationMetrics,
    SourceReliability,
    QualitySignals,
    ContributorCalibration,
    QualityEngineConfig,
    QualitySignalEngine,
    get_quality_signal_engine,
)
from aragora.knowledge.mound.ops.composite_analytics import (
    SLOStatus,
    BottleneckSeverity,
    OptimizationType,
    AdapterMetrics,
    SLOConfig,
    SLOResult,
    BottleneckAnalysis,
    OptimizationRecommendation,
    CompositeMetrics,
    SyncResultInput,
    CompositeAnalytics,
    get_composite_analytics,
)
from aragora.knowledge.mound.ops.contradiction import ValidatorVote

__all__ = [
    "StalenessOperationsMixin",
    "CultureOperationsMixin",
    "SyncOperationsMixin",
    "GlobalKnowledgeMixin",
    "KnowledgeSharingMixin",
    "KnowledgeFederationMixin",
    "FederationMode",
    "SyncScope",
    "FederatedRegion",
    "SyncResult",
    "SYSTEM_WORKSPACE_ID",
    # Dedup
    "DedupOperationsMixin",
    "DuplicateCluster",
    "DuplicateMatch",
    "DedupReport",
    "MergeResult",
    # Pruning
    "PruningOperationsMixin",
    "PruningPolicy",
    "PrunableItem",
    "PruneResult",
    "PruneHistory",
    "PruningAction",
    # Auto-curation (Phase 4)
    "AutoCurationMixin",
    "CurationPolicy",
    "CurationCandidate",
    "CurationResult",
    "CurationHistory",
    "CurationAction",
    "QualityScore",
    "TierLevel",
    # Phase A2: Contradiction Detection
    "ContradictionOperationsMixin",
    "ContradictionDetector",
    "Contradiction",
    "ContradictionReport",
    "ContradictionConfig",
    "ContradictionType",
    "ResolutionStrategy",
    "get_contradiction_detector",
    # Phase A2: Confidence Decay
    "ConfidenceDecayMixin",
    "ConfidenceDecayManager",
    "ConfidenceAdjustment",
    "DecayReport",
    "DecayConfig",
    "DecayModel",
    "ConfidenceEvent",
    "get_decay_manager",
    # Phase A2: Governance
    "GovernanceMixin",
    "RBACManager",
    "AuditTrail",
    "Role",
    "RoleAssignment",
    "Permission",
    "BuiltinRole",
    "BUILTIN_ROLES",
    "AuditAction",
    "AuditEntry",
    "get_rbac_manager",
    "get_audit_trail",
    # Phase A2: Analytics
    "AnalyticsMixin",
    "KnowledgeAnalytics",
    "DomainCoverage",
    "CoverageReport",
    "UsageEvent",
    "UsageEventType",
    "ItemUsageStats",
    "UsageReport",
    "QualitySnapshot",
    "QualityTrend",
    "GrowthMetrics",
    "get_knowledge_analytics",
    # Phase A2: Knowledge Extraction
    "ExtractionMixin",
    "DebateKnowledgeExtractor",
    "ExtractedClaim",
    "ExtractedRelationship",
    "ExtractionResult",
    "ExtractionConfig",
    "ExtractionType",
    "ConfidenceSource",
    "get_debate_extractor",
    # Phase A3: Adapter Fusion Protocol
    "FusionStrategy",
    "ConflictResolution",
    "AdapterValidation",
    "FusedValidation",
    "FusionConfig",
    "FusionCoordinator",
    "get_fusion_coordinator",
    # Phase A3: Calibration Fusion
    "CalibrationFusionStrategy",
    "AgentPrediction",
    "CalibrationConsensus",
    "CalibrationFusionConfig",
    "CalibrationFusionEngine",
    "get_calibration_fusion_engine",
    # Phase A3: Multi-Party Validation
    "ValidationVoteType",
    "ValidationConsensusStrategy",
    "ValidationState",
    "ValidationVote",
    "ValidationRequest",
    "ValidationResult",
    "EscalationResult",
    "ValidatorConfig",
    "MultiPartyValidator",
    "get_multi_party_validator",
    "ValidatorVote",  # From contradiction for validator voting
    # Phase A3: Quality Signals
    "QualityDimension",
    "OverconfidenceLevel",
    "QualityTier",
    "CalibrationMetrics",
    "SourceReliability",
    "QualitySignals",
    "ContributorCalibration",
    "QualityEngineConfig",
    "QualitySignalEngine",
    "get_quality_signal_engine",
    # Phase A3: Composite Analytics
    "SLOStatus",
    "BottleneckSeverity",
    "OptimizationType",
    "AdapterMetrics",
    "SLOConfig",
    "SLOResult",
    "BottleneckAnalysis",
    "OptimizationRecommendation",
    "CompositeMetrics",
    "SyncResultInput",
    "CompositeAnalytics",
    "get_composite_analytics",
]
