"""PR intelligence brief — heterogeneous review protocol and brief schema.

Implements the type contracts that #6307 (receipt schema), #6304 (UI), and
#6305 (cost controls) all import. This module deliberately contains only
data shapes and enums — no behavior, no I/O, no orchestration. Behavior
ships in successor PRs against the same package.

Design brief: docs/plans/2026-04-19-pr-intelligence-brief.md
Tracking: #6306
"""

from aragora.review.builder import (
    PanelVote,
    build_brief,
    compute_packet_sha,
)
from aragora.review.protocol import (
    ADVISORY_NOTE,
    DissentingView,
    DissentPosition,
    PRReviewProtocol,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SynthesisPolicy,
)
from aragora.review.policy import (
    BudgetHeadroom,
    BudgetScope,
    CostMeter,
    DepthTrigger,
    ReviewBudget,
    ReviewDepth,
    ReviewPolicy,
    ReviewPolicyDecision,
    RiskClass,
)
from aragora.review.receipt import (
    BriefReceipt,
    EvidenceKind,
    EvidenceRef,
    SettlementAction,
    SettlementLinkage,
    ValidationKind,
    ValidationRef,
    ValidationResult,
)
from aragora.review.provider_slots import (
    ProviderCandidateCheck,
    ProviderSlotAvailabilitySummary,
    ProviderSlotDefinition,
    ProviderSlotResolution,
    ProviderSlotResolver,
)
from aragora.review.reviewer_output import (
    FindingCategory,
    FindingSeverity,
    REVIEWER_OUTPUT_SCHEMA_VERSION,
    ReviewerFinding,
    ReviewerOutput,
    validate_reviewer_outputs,
)
from aragora.review.invalidation import (
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_REVERT_WINDOW_DAYS,
    DEFAULT_SAFETY_MARGIN,
    DEFAULT_MINIMUM_MEANINGFUL_RATE,
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_REOPENED_PR,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    INVALIDATION_ROLLBACK,
    INVALIDATION_SIGNALS,
    BaselineMeasurement,
    InvalidatedDecision,
    ThresholdProposal,
    classify_invalidation,
    compute_baseline,
    derive_threshold,
    is_invalidated,
)

__all__ = [
    "ADVISORY_NOTE",
    "BaselineMeasurement",
    "BriefReceipt",
    "BudgetHeadroom",
    "BudgetScope",
    "CostMeter",
    "DEFAULT_BASELINE_WINDOW_DAYS",
    "DEFAULT_MIN_BASELINE_SAMPLES",
    "DEFAULT_MINIMUM_MEANINGFUL_RATE",
    "DEFAULT_REVERT_WINDOW_DAYS",
    "DEFAULT_SAFETY_MARGIN",
    "DepthTrigger",
    "DissentingView",
    "DissentPosition",
    "EvidenceKind",
    "EvidenceRef",
    "FindingCategory",
    "FindingSeverity",
    "INVALIDATION_HUMAN_OVERRIDE_REDO",
    "INVALIDATION_POST_MERGE_INCIDENT",
    "INVALIDATION_REOPENED_PR",
    "INVALIDATION_REVERT_WITHIN_WINDOW",
    "INVALIDATION_ROLLBACK",
    "INVALIDATION_SIGNALS",
    "InvalidatedDecision",
    "PRReviewProtocol",
    "ProviderCandidateCheck",
    "ProviderSlotAvailabilitySummary",
    "ProviderSlotDefinition",
    "ProviderSlotResolution",
    "ProviderSlotResolver",
    "PanelVote",
    "REVIEWER_OUTPUT_SCHEMA_VERSION",
    "Recommendation",
    "ReviewBrief",
    "ReviewBudget",
    "ReviewDepth",
    "ReviewerFinding",
    "ReviewerOutput",
    "ReviewPolicy",
    "ReviewPolicyDecision",
    "ReviewRole",
    "RiskClass",
    "RoleFinding",
    "SettlementAction",
    "SettlementLinkage",
    "SynthesisPolicy",
    "ThresholdProposal",
    "ValidationKind",
    "ValidationRef",
    "ValidationResult",
    "build_brief",
    "classify_invalidation",
    "compute_baseline",
    "compute_packet_sha",
    "derive_threshold",
    "is_invalidated",
    "validate_reviewer_outputs",
]
