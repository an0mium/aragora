"""Triage observability package — rolling-window metric computation.

Implements the measurement substrate for Commitment 5 of the Aragora
thesis (docs/THESIS.md), which requires the triage layer to emit per
rolling window:

  - escalation rate
  - auto-handle override rate
  - human-override-outcome correlation
  - time-per-settlement

The package is split into two layers:

``metrics`` — pure-function aggregation over ``TriageDecisionEvent``
sequences. No I/O, no database, no filesystem. Testable with synthetic
event lists.

``event_source`` — adapter that reads existing brief receipts,
settlement receipts, and PDB brief index events from disk and yields
``TriageDecisionEvent`` instances. Does NOT modify receipt schemas.

The HTTP surface for these metrics lives in
:mod:`aragora.server.handlers.triage_metrics` (gap #6373).
"""

from __future__ import annotations

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

__all__ = [
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
    "AutoHandleStoreError",
    "TriageDecisionEvent",
    "TriageWindowMetrics",
    "auto_handle_decision_id",
    "compute_window",
    "detect_drift",
    "fingerprint_admin_merge_class",
    "fingerprint_low_risk_class",
]
