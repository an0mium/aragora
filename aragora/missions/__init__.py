"""Native mission orchestrator (Phase A spine).

A survivable, goal-driven mission engine: the orchestrator is *stateless between
ticks* and reconstructs everything from disk, so it resumes cleanly after a
402 / crash / ``kill -9``. See docs/plans/2026-06-25-native-mission-orchestrator-spec.md.

Public surface::

    from aragora.missions import MissionState, MissionOrchestrator, Handoff
"""

from .ledger import Constraint, Ledger, Lease, select_for
from .live_gate import LiveBossLoopGate
from .orchestrator import Handoff, MissionOrchestrator
from .reconcile import (
    AdmissionDecision,
    AdmissionPolicy,
    ArtifactCategory,
    ClassifiedArtifact,
    ReconcileMode,
    ReconcileReport,
    WorkArtifact,
    apply_validation_result,
    classify_artifact,
    inject_validation_features,
    write_operator_receipt,
)
from .runtime import MissionRuntimeConfig
from .state import (
    Feature,
    MissionOwnershipError,
    MissionState,
    Status,
    mission_owner_lock,
)
from .swarm import SwarmResult, reconcile_from_ledger, run_worker

__all__ = [
    "Constraint",
    "Feature",
    "Handoff",
    "Ledger",
    "Lease",
    "LiveBossLoopGate",
    "MissionOrchestrator",
    "MissionOwnershipError",
    "MissionRuntimeConfig",
    "MissionState",
    "AdmissionDecision",
    "AdmissionPolicy",
    "ArtifactCategory",
    "ClassifiedArtifact",
    "ReconcileMode",
    "ReconcileReport",
    "Status",
    "SwarmResult",
    "WorkArtifact",
    "apply_validation_result",
    "classify_artifact",
    "inject_validation_features",
    "mission_owner_lock",
    "reconcile_from_ledger",
    "run_worker",
    "select_for",
    "write_operator_receipt",
]
