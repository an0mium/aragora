"""Native mission orchestrator (Phase A spine).

A survivable, goal-driven mission engine: the orchestrator is *stateless between
ticks* and reconstructs everything from disk, so it resumes cleanly after a
402 / crash / ``kill -9``. See docs/plans/2026-06-25-native-mission-orchestrator-spec.md.

Public surface::

    from aragora.missions import MissionState, MissionOrchestrator, Handoff
"""

from .ledger import Constraint, Ledger, Lease, select_for
from .orchestrator import Handoff, MissionOrchestrator
from .state import Feature, MissionState, Status
from .swarm import SwarmResult, run_worker

__all__ = [
    "Constraint",
    "Feature",
    "Handoff",
    "Ledger",
    "Lease",
    "MissionOrchestrator",
    "MissionState",
    "Status",
    "SwarmResult",
    "run_worker",
    "select_for",
]
