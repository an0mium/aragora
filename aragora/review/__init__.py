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

__all__ = [
    "ADVISORY_NOTE",
    "BriefReceipt",
    "BudgetHeadroom",
    "BudgetScope",
    "CostMeter",
    "DepthTrigger",
    "DissentingView",
    "DissentPosition",
    "EvidenceKind",
    "EvidenceRef",
    "FindingCategory",
    "FindingSeverity",
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
    "ValidationKind",
    "ValidationRef",
    "ValidationResult",
    "build_brief",
    "compute_packet_sha",
    "validate_reviewer_outputs",
]
