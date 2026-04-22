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

from aragora.triage.metrics import (
    MIN_EVENTS_FOR_METRICS,
    TriageDecisionEvent,
    TriageWindowMetrics,
    compute_window,
    detect_drift,
)

__all__ = [
    "MIN_EVENTS_FOR_METRICS",
    "TriageDecisionEvent",
    "TriageWindowMetrics",
    "compute_window",
    "detect_drift",
]
