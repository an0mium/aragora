"""Triage observability and issue-triage utilities.

This package historically holds the measurement substrate for
Commitment 5 of the Aragora thesis (rolling-window triage metrics and
auto-handle calibration). It now also hosts a calibration-only
multi-model GitHub issue triage library used by
``scripts/triage_issues_via_debate.py`` to evaluate the repository's
own issue backlog with a heterogeneous frontier panel.

Two distinct subsystems live here; both are intentionally independent
modules so neither imports the other implicitly.

Observability submodules
------------------------
``metrics`` -- pure-function aggregation over ``TriageDecisionEvent``
sequences. No I/O, no database, no filesystem. Testable with synthetic
event lists.

``event_source`` -- adapter that reads existing brief receipts,
settlement receipts, and PDB brief index events from disk and yields
``TriageDecisionEvent`` instances. Does NOT modify receipt schemas.

``auto_handle_calibration`` -- per-class auto-handle gate decisions
and drift detection.

Issue-triage submodules (calibration v1)
----------------------------------------
``evidence`` -- gather GitHub + repo evidence before any model call.

``receipts`` -- persist per-model and aggregate audit artifacts.

``issue_evaluator`` -- panel construction, prompt assembly, aggregation,
the main ``evaluate_issue`` entry point.

The HTTP surface for observability metrics lives in
:mod:`aragora.server.handlers.triage_metrics` (gap #6373). The CLI
surface for the issue-triage library lives in
``scripts/triage_issues_via_debate.py``.
"""

from __future__ import annotations

# Observability surface (unchanged, mirrors prior contract).
from aragora.triage.auto_handle_calibration import (
    AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED,
    AUTO_HANDLE_PATH_FIRE_AND_FORGET,
    OUTCOME_HUMAN_OVERRIDE,
    OUTCOME_INCIDENT,
    OUTCOME_REVERT,
    OUTCOME_SUCCESS,
    AutoHandleCalibrationStore,
    AutoHandleClassSummary,
    AutoHandleDriftAlert,
    AutoHandleGateDecision,
    AutoHandleStoreError,
    auto_handle_decision_id,
    fingerprint_admin_merge_class,
    fingerprint_low_risk_class,
)
from aragora.triage.metrics import (
    MIN_EVENTS_FOR_METRICS,
    TriageDecisionEvent,
    TriageWindowMetrics,
    compute_window,
    detect_drift,
)

# Issue-triage calibration surface (new).
from aragora.triage.evidence import (
    IssueEvidence,
    IssueRecord,
    gather_evidence,
    is_automation_generated,
)
from aragora.triage.issue_evaluator import (
    AUTOMATION_VALUE_VALUES,
    CONFIDENCE_CLASSES,
    DEFAULT_PANEL,
    PANEL_PROMPT_RUBRIC,
    VERDICT_CATEGORIES,
    AggregateVerdict,
    FounderRecommendation,
    PanelMember,
    PerModelVerdict,
    aggregate_verdicts,
    build_panel,
    build_panel_prompt,
    estimate_cost_usd,
    evaluate_issue,
    parse_model_response,
)
from aragora.triage.receipts import (
    RECEIPT_SCHEMA_VERSION,
    IssueDebateReceipt,
    write_jsonl_receipt,
    write_markdown_report,
)

__all__ = [
    # Observability (preserved)
    "AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED",
    "AUTO_HANDLE_PATH_FIRE_AND_FORGET",
    "MIN_EVENTS_FOR_METRICS",
    "OUTCOME_HUMAN_OVERRIDE",
    "OUTCOME_INCIDENT",
    "OUTCOME_REVERT",
    "OUTCOME_SUCCESS",
    "AutoHandleCalibrationStore",
    "AutoHandleClassSummary",
    "AutoHandleDriftAlert",
    "AutoHandleGateDecision",
    "AutoHandleStoreError",
    "TriageDecisionEvent",
    "TriageWindowMetrics",
    "auto_handle_decision_id",
    "compute_window",
    "detect_drift",
    "fingerprint_admin_merge_class",
    "fingerprint_low_risk_class",
    # Issue-triage calibration (new)
    "AUTOMATION_VALUE_VALUES",
    "AggregateVerdict",
    "CONFIDENCE_CLASSES",
    "DEFAULT_PANEL",
    "FounderRecommendation",
    "IssueDebateReceipt",
    "IssueEvidence",
    "IssueRecord",
    "PANEL_PROMPT_RUBRIC",
    "PanelMember",
    "PerModelVerdict",
    "RECEIPT_SCHEMA_VERSION",
    "VERDICT_CATEGORIES",
    "aggregate_verdicts",
    "build_panel",
    "build_panel_prompt",
    "estimate_cost_usd",
    "evaluate_issue",
    "gather_evidence",
    "is_automation_generated",
    "parse_model_response",
    "write_jsonl_receipt",
    "write_markdown_report",
]
