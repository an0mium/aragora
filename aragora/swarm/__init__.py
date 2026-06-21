"""Swarm Commander: interrogate -> spec -> dispatch -> merge -> report.

The swarm module provides a user-facing wrapper around Aragora's existing
orchestration infrastructure. It adds an interrogation phase (gathering
requirements from non-developer users) and a reporting phase (explaining
results in plain English).

Usage:
    from aragora.swarm import SwarmCommander, SwarmSpec, SwarmReport

    # Full lifecycle
    commander = SwarmCommander()
    report = await commander.run("Make the dashboard faster")
    print(report.to_plain_text())

    # From pre-built spec
    spec = SwarmSpec.from_yaml(Path("my-spec.yaml").read_text())
    report = await commander.run_from_spec(spec)

    # Dry run (spec only, no dispatch)
    spec = await commander.dry_run("Improve test coverage")
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "CampaignExecutionState": ("aragora.swarm.campaign", "CampaignExecutionState"),
    "CampaignManifest": ("aragora.swarm.campaign", "CampaignManifest"),
    "CampaignProject": ("aragora.swarm.campaign", "CampaignProject"),
    "CampaignReviewGate": ("aragora.swarm.campaign", "CampaignReviewGate"),
    "CampaignRunOutcome": ("aragora.swarm.campaign", "CampaignRunOutcome"),
    "CampaignStopReason": ("aragora.swarm.campaign", "CampaignStopReason"),
    "InterrogatorConfig": ("aragora.swarm.config", "InterrogatorConfig"),
    "LaunchConfig": ("aragora.swarm.worker_launcher", "LaunchConfig"),
    "SwarmCommander": ("aragora.swarm.commander", "SwarmCommander"),
    "SwarmCommanderConfig": ("aragora.swarm.config", "SwarmCommanderConfig"),
    "TrancheArtifactStore": ("aragora.swarm.tranche", "TrancheArtifactStore"),
    "TrancheExecutor": ("aragora.swarm.tranche", "TrancheExecutor"),
    "TrancheGate": ("aragora.swarm.tranche", "TrancheGate"),
    "TrancheInspector": ("aragora.swarm.tranche", "TrancheInspector"),
    "TrancheLane": ("aragora.swarm.tranche", "TrancheLane"),
    "TrancheLaneArtifact": ("aragora.swarm.tranche", "TrancheLaneArtifact"),
    "TrancheManifest": ("aragora.swarm.tranche", "TrancheManifest"),
    "TranchePlanner": ("aragora.swarm.tranche", "TranchePlanner"),
    "SwarmReconciler": ("aragora.swarm.reconciler", "SwarmReconciler"),
    "SwarmReconcilerConfig": ("aragora.swarm.reconciler", "SwarmReconcilerConfig"),
    "SupervisorRun": ("aragora.swarm.supervisor", "SupervisorRun"),
    "SwarmReport": ("aragora.swarm.reporter", "SwarmReport"),
    "SwarmReporter": ("aragora.swarm.reporter", "SwarmReporter"),
    "SwarmApprovalPolicy": ("aragora.swarm.supervisor", "SwarmApprovalPolicy"),
    "SwarmSpec": ("aragora.swarm.spec", "SwarmSpec"),
    "SwarmSupervisor": ("aragora.swarm.supervisor", "SwarmSupervisor"),
    "WorkerLauncher": ("aragora.swarm.worker_launcher", "WorkerLauncher"),
    "WorkerProcess": ("aragora.swarm.worker_launcher", "WorkerProcess"),
    "LaneRunState": ("aragora.swarm.tranche_state", "LaneRunState"),
    "TrancheRunState": ("aragora.swarm.tranche_state", "TrancheRunState"),
    "load_tranche_manifest": ("aragora.swarm.tranche", "load_tranche_manifest"),
    "review_lane": ("aragora.swarm.tranche_review", "review_lane"),
    "save_tranche_manifest": ("aragora.swarm.tranche", "save_tranche_manifest"),
    "submit_intake_bundle": ("aragora.swarm.tranche_submit", "submit_intake_bundle"),
    "assess_lane_integration": ("aragora.swarm.tranche_integrate", "assess_lane_integration"),
}

__all__ = [
    "CampaignExecutionState",
    "CampaignManifest",
    "CampaignProject",
    "CampaignReviewGate",
    "CampaignRunOutcome",
    "CampaignStopReason",
    "InterrogatorConfig",
    "LaunchConfig",
    "SwarmCommander",
    "SwarmCommanderConfig",
    "TrancheArtifactStore",
    "TrancheExecutor",
    "TrancheGate",
    "TrancheInspector",
    "TrancheLane",
    "TrancheLaneArtifact",
    "TrancheManifest",
    "TranchePlanner",
    "SwarmReconciler",
    "SwarmReconcilerConfig",
    "SupervisorRun",
    "SwarmReport",
    "SwarmReporter",
    "SwarmApprovalPolicy",
    "SwarmSpec",
    "SwarmSupervisor",
    "WorkerLauncher",
    "WorkerProcess",
    "LaneRunState",
    "TrancheRunState",
    "load_tranche_manifest",
    "review_lane",
    "save_tranche_manifest",
    "submit_intake_bundle",
    "assess_lane_integration",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
