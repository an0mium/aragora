"""Native mission orchestrator (Phase A spine).

A survivable, goal-driven mission engine: the orchestrator is *stateless between
ticks* and reconstructs everything from disk, so it resumes cleanly after a
402 / crash / ``kill -9``. See docs/plans/2026-06-25-native-mission-orchestrator-spec.md.

Public surface::

    from aragora.missions import MissionState, MissionOrchestrator, Handoff
"""

from .orchestrator import Handoff, MissionOrchestrator
from .state import Feature, MissionState, Status

__all__ = [
    "Feature",
    "Handoff",
    "MissionOrchestrator",
    "MissionState",
    "Status",
]
