"""PR intelligence brief — heterogeneous review protocol and brief schema.

Implements the type contracts that #6307 (receipt schema), #6304 (UI), and
#6305 (cost controls) all import. This module deliberately contains only
data shapes and enums — no behavior, no I/O, no orchestration. Behavior
ships in successor PRs against the same package.

Design brief: docs/plans/2026-04-19-pr-intelligence-brief.md
Tracking: #6306
"""

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

__all__ = [
    "ADVISORY_NOTE",
    "BriefReceipt",
    "DissentingView",
    "DissentPosition",
    "EvidenceKind",
    "EvidenceRef",
    "PRReviewProtocol",
    "Recommendation",
    "ReviewBrief",
    "ReviewRole",
    "RoleFinding",
    "SettlementAction",
    "SettlementLinkage",
    "SynthesisPolicy",
    "ValidationKind",
    "ValidationRef",
    "ValidationResult",
]
