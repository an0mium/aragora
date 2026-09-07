"""Deprecated re-export shim for telemetry research events.

The originals now live in :mod:`aragora.observability.research_events`.
This module re-exports them so the legacy ``aragora.telemetry.research_events``
import path keeps working for one release.
"""

import warnings

from aragora.observability.research_events import (
    TelemetryEvent,
    TelemetryEventType,
    ascot_fragility_event,
    claim_verified_event,
    early_termination_event,
    graph_expansion_event,
    integration_error_event,
    muse_calculation_event,
    prm_error_detected_event,
    prm_step_verified_event,
    role_assignment_event,
    routing_decision_event,
    stability_check_event,
    team_composed_event,
)

warnings.warn(
    "aragora.telemetry.research_events is deprecated; import from "
    "aragora.observability.research_events instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TelemetryEvent",
    "TelemetryEventType",
    "ascot_fragility_event",
    "claim_verified_event",
    "early_termination_event",
    "graph_expansion_event",
    "integration_error_event",
    "muse_calculation_event",
    "prm_error_detected_event",
    "prm_step_verified_event",
    "role_assignment_event",
    "routing_decision_event",
    "stability_check_event",
    "team_composed_event",
]
