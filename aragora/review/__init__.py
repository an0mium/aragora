"""PR intelligence brief public exports.

The package root intentionally avoids eager imports. Readiness helpers import
small review submodules during merge-gate checks, and importing this package
must not pull provider registries, calibration stores, or live-review stacks
unless a caller actually asks for those objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ADVISORY_NOTE": ("aragora.review.protocol", "ADVISORY_NOTE"),
    "BaselineMeasurement": ("aragora.review.invalidation", "BaselineMeasurement"),
    "BriefReceipt": ("aragora.review.receipt", "BriefReceipt"),
    "BudgetHeadroom": ("aragora.review.policy", "BudgetHeadroom"),
    "BudgetScope": ("aragora.review.policy", "BudgetScope"),
    "CostMeter": ("aragora.review.policy", "CostMeter"),
    "DEFAULT_BASELINE_WINDOW_DAYS": (
        "aragora.review.invalidation",
        "DEFAULT_BASELINE_WINDOW_DAYS",
    ),
    "DEFAULT_MIN_BASELINE_SAMPLES": (
        "aragora.review.invalidation",
        "DEFAULT_MIN_BASELINE_SAMPLES",
    ),
    "DEFAULT_MINIMUM_MEANINGFUL_RATE": (
        "aragora.review.invalidation",
        "DEFAULT_MINIMUM_MEANINGFUL_RATE",
    ),
    "DEFAULT_REVERT_WINDOW_DAYS": ("aragora.review.invalidation", "DEFAULT_REVERT_WINDOW_DAYS"),
    "DEFAULT_SAFETY_MARGIN": ("aragora.review.invalidation", "DEFAULT_SAFETY_MARGIN"),
    "DEFAULT_SUPPORT_TARGETS": (
        "aragora.review.threshold_recalibration",
        "DEFAULT_SUPPORT_TARGETS",
    ),
    "DEFAULT_THRESHOLD_RECEIPT_DIR": (
        "aragora.review.threshold_recalibration",
        "DEFAULT_THRESHOLD_RECEIPT_DIR",
    ),
    "DepthTrigger": ("aragora.review.policy", "DepthTrigger"),
    "DissentingView": ("aragora.review.protocol", "DissentingView"),
    "DissentPosition": ("aragora.review.protocol", "DissentPosition"),
    "EvidenceKind": ("aragora.review.receipt", "EvidenceKind"),
    "EvidenceRef": ("aragora.review.receipt", "EvidenceRef"),
    "FindingCategory": ("aragora.review.reviewer_output", "FindingCategory"),
    "FindingSeverity": ("aragora.review.reviewer_output", "FindingSeverity"),
    "INVALIDATION_HUMAN_OVERRIDE_REDO": (
        "aragora.review.invalidation",
        "INVALIDATION_HUMAN_OVERRIDE_REDO",
    ),
    "INVALIDATION_POST_MERGE_INCIDENT": (
        "aragora.review.invalidation",
        "INVALIDATION_POST_MERGE_INCIDENT",
    ),
    "INVALIDATION_REOPENED_PR": ("aragora.review.invalidation", "INVALIDATION_REOPENED_PR"),
    "INVALIDATION_REVERT_WITHIN_WINDOW": (
        "aragora.review.invalidation",
        "INVALIDATION_REVERT_WITHIN_WINDOW",
    ),
    "INVALIDATION_ROLLBACK": ("aragora.review.invalidation", "INVALIDATION_ROLLBACK"),
    "INVALIDATION_SIGNALS": ("aragora.review.invalidation", "INVALIDATION_SIGNALS"),
    "INSUFFICIENCY_RECEIPT_SCHEMA_VERSION": (
        "aragora.review.threshold_recalibration",
        "INSUFFICIENCY_RECEIPT_SCHEMA_VERSION",
    ),
    "InsufficiencyReceipt": ("aragora.review.threshold_recalibration", "InsufficiencyReceipt"),
    "InvalidationEventSource": (
        "aragora.review.threshold_recalibration",
        "InvalidationEventSource",
    ),
    "InvalidationRecalibrationSample": (
        "aragora.review.threshold_recalibration",
        "InvalidationRecalibrationSample",
    ),
    "InvalidatedDecision": ("aragora.review.invalidation", "InvalidatedDecision"),
    "PRReviewProtocol": ("aragora.review.protocol", "PRReviewProtocol"),
    "PanelVote": ("aragora.review.builder", "PanelVote"),
    "ProviderCandidateCheck": ("aragora.review.provider_slots", "ProviderCandidateCheck"),
    "ProviderSlotAvailabilitySummary": (
        "aragora.review.provider_slots",
        "ProviderSlotAvailabilitySummary",
    ),
    "ProviderSlotDefinition": ("aragora.review.provider_slots", "ProviderSlotDefinition"),
    "ProviderSlotResolution": ("aragora.review.provider_slots", "ProviderSlotResolution"),
    "ProviderSlotResolver": ("aragora.review.provider_slots", "ProviderSlotResolver"),
    "REVIEWER_OUTPUT_SCHEMA_VERSION": (
        "aragora.review.reviewer_output",
        "REVIEWER_OUTPUT_SCHEMA_VERSION",
    ),
    "Recommendation": ("aragora.review.protocol", "Recommendation"),
    "RecalibrationReceipt": ("aragora.review.threshold_recalibration", "RecalibrationReceipt"),
    "ReviewBrief": ("aragora.review.protocol", "ReviewBrief"),
    "ReviewBudget": ("aragora.review.policy", "ReviewBudget"),
    "ReviewDepth": ("aragora.review.policy", "ReviewDepth"),
    "ReviewPolicy": ("aragora.review.policy", "ReviewPolicy"),
    "ReviewPolicyDecision": ("aragora.review.policy", "ReviewPolicyDecision"),
    "ReviewQueueInvalidationEventSource": (
        "aragora.review.invalidation_event_source",
        "ReviewQueueInvalidationEventSource",
    ),
    "ReviewRole": ("aragora.review.protocol", "ReviewRole"),
    "ReviewerFinding": ("aragora.review.reviewer_output", "ReviewerFinding"),
    "ReviewerOutput": ("aragora.review.reviewer_output", "ReviewerOutput"),
    "RiskClass": ("aragora.review.policy", "RiskClass"),
    "RoleFinding": ("aragora.review.protocol", "RoleFinding"),
    "SettlementAction": ("aragora.review.receipt", "SettlementAction"),
    "SettlementLinkage": ("aragora.review.receipt", "SettlementLinkage"),
    "SynthesisPolicy": ("aragora.review.protocol", "SynthesisPolicy"),
    "THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION": (
        "aragora.review.threshold_recalibration",
        "THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION",
    ),
    "ThresholdProposal": ("aragora.review.invalidation", "ThresholdProposal"),
    "ThresholdRecalibrationScheduler": (
        "aragora.review.threshold_recalibration",
        "ThresholdRecalibrationScheduler",
    ),
    "ThresholdUpdateReceipt": (
        "aragora.review.threshold_recalibration",
        "ThresholdUpdateReceipt",
    ),
    "ValidationKind": ("aragora.review.receipt", "ValidationKind"),
    "ValidationRef": ("aragora.review.receipt", "ValidationRef"),
    "ValidationResult": ("aragora.review.receipt", "ValidationResult"),
    "build_brief": ("aragora.review.builder", "build_brief"),
    "classify_invalidation": ("aragora.review.invalidation", "classify_invalidation"),
    "compute_baseline": ("aragora.review.invalidation", "compute_baseline"),
    "compute_insufficiency_receipt_id": (
        "aragora.review.threshold_recalibration",
        "compute_insufficiency_receipt_id",
    ),
    "compute_packet_sha": ("aragora.review.builder", "compute_packet_sha"),
    "compute_threshold_update_receipt_id": (
        "aragora.review.threshold_recalibration",
        "compute_threshold_update_receipt_id",
    ),
    "count_decisions_from_settlement_receipts": (
        "aragora.review.invalidation_event_source",
        "count_decisions_from_settlement_receipts",
    ),
    "derive_threshold": ("aragora.review.invalidation", "derive_threshold"),
    "is_invalidated": ("aragora.review.invalidation", "is_invalidated"),
    "iter_invalidations_from_calibration_store": (
        "aragora.review.invalidation_event_source",
        "iter_invalidations_from_calibration_store",
    ),
    "iter_invalidations_from_settlement_receipts": (
        "aragora.review.invalidation_event_source",
        "iter_invalidations_from_settlement_receipts",
    ),
    "measure_baseline_from_stores": (
        "aragora.review.invalidation_event_source",
        "measure_baseline_from_stores",
    ),
    "resolve_review_queue_root": (
        "aragora.review.invalidation_event_source",
        "resolve_review_queue_root",
    ),
    "validate_reviewer_outputs": ("aragora.review.reviewer_output", "validate_reviewer_outputs"),
    "write_insufficiency_receipt": (
        "aragora.review.threshold_recalibration",
        "write_insufficiency_receipt",
    ),
    "write_recalibration_receipt": (
        "aragora.review.threshold_recalibration",
        "write_recalibration_receipt",
    ),
    "write_threshold_update_receipt": (
        "aragora.review.threshold_recalibration",
        "write_threshold_update_receipt",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# Golden API collision guard (issue #8780)
#
# This subpackage shares its name with the golden callable
# ``aragora.golden.review`` that ``aragora/__init__.py`` exports lazily via
# ``_EXPORT_MAP``. When this subpackage is imported, the import system binds
# the module object onto the ``aragora`` package, shadowing the golden callable.
# Making the module itself callable keeps ``aragora.review(...)`` working in
# every import order while preserving normal module semantics.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
from typing import Any as _CallableAny


class _CallableReviewModule(_types.ModuleType):
    """Module subclass forwarding calls to :func:`aragora.golden.review`."""

    def __call__(self, *args: _CallableAny, **kwargs: _CallableAny) -> _CallableAny:
        from aragora.golden import review as _golden_review

        return _golden_review(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableReviewModule
