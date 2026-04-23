"""Epistemic CI and crux-engine helpers (DIC-13..22 + DIC-25/26 tranche).

Exposes:
- DIC-14: executable claim verification (:class:`ClaimVerifier`)
- DIC-16: signed CruxReceipt for crux-finder debate runs
  (:class:`CruxEntry`, :class:`CruxReceipt`, :func:`build_crux_receipt`)
- DIC-17: follow-up-issue bridge for load-bearing cruxes and
  sharply-losing claims (:class:`FollowupProposal`,
  :func:`propose_followup_for_crux`, :func:`propose_followup_for_cruxset`,
  :func:`propose_followup_for_failed_claim`)
- DIC-18: organizational truth map report (:class:`OrgTruthMapReport`,
  :func:`build_truth_map`, :func:`build_truth_map_from_manifests`)
- DIC-20: epistemic decay monitor (:class:`DecaySignal`,
  :class:`DecayReason`, :func:`evaluate_unit`)
- DIC-21: fail-closed quarantine policy (:class:`QuarantineDecision`,
  :class:`QuarantinePolicy`, :func:`apply_quarantine_policy`,
  :func:`quarantine_policy_enabled`)
- DIC-22: bounded repair-spec producer (:class:`RepairSpec`,
  :func:`propose_repair`, :func:`repair_pipeline_enabled`)
- DIC-25: adversarial world-state stress-test (:class:`StressPerturbation`,
  :class:`FragilityReport`, :class:`StressTestResult`,
  :func:`run_stress_test`, :func:`stress_test_enabled`)
  Flag gate: ``ARAGORA_STRESS_TEST_ENABLED`` (default off).
- DIC-26: belief coherence monitor (:class:`BeliefEntry`,
  :class:`CoherenceReport`, :func:`scan_coherence`)

See ``docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md`` for the full
DIC-13..22 + DIC-23..28 sequence and ``docs/status/NEXT_STEPS_CANONICAL.md``
for the queue-governance activation gate.
"""

from __future__ import annotations

import os

from .arbitration import (
    PERSISTENT_CRUX_MIN_CONSECUTIVE,
    PERSISTENT_CRUX_MIN_SCORE,
    DEFAULT_EXPIRY_DAYS,
    ArbitrationSide,
    CruxArbitration,
    CruxArbitrationReversal,
    PersistentCrux,
    build_arbitration,
    build_reversal,
    crux_arbitration_enabled,
)
from .claim_verifier import ClaimResult, ClaimStatus, ClaimVerifier
from .coherence import (
    BeliefEntry,
    CoherenceIssue,
    CoherenceReport,
    IncoherenceKind,
    coherence_monitor_enabled,
    from_belief_node,
    scan_coherence,
)
from .crux_receipt import (
    CruxEntry,
    CruxReceipt,
    build_crux_receipt,
    crux_receipt_enabled,
    enable_crux_receipt,
)
from .decay_monitor import DecayReason, DecaySignal, evaluate_unit
from .followup import (
    DEFAULT_CRUX_LOAD_BEARING_THRESHOLD,
    DEFAULT_DELTA_LOSS_THRESHOLD,
    FollowupProposal,
    propose_followup_for_crux,
    propose_followup_for_cruxset,
    propose_followup_for_failed_claim,
)
from .quarantine_policy import (
    QuarantineDecision,
    QuarantinePolicy,
    apply_quarantine_policy,
    quarantine_policy_enabled,
)
from .repair import (
    RepairSpec,
    enable_repair_pipeline,
    propose_repair,
    repair_pipeline_enabled,
)
from .stress_test import (
    FragilityReport,
    StressPerturbation,
    StressTestResult,
    run_stress_test,
    stress_test_enabled,
)
from .truth_map import (
    OrgTruthMapReport,
    build_truth_map,
    build_truth_map_from_manifests,
)

__all__ = [
    "ArbitrationSide",
    "BeliefEntry",
    "ClaimResult",
    "ClaimStatus",
    "ClaimVerifier",
    "CruxArbitration",
    "CruxArbitrationReversal",
    "DEFAULT_EXPIRY_DAYS",
    "PERSISTENT_CRUX_MIN_CONSECUTIVE",
    "PERSISTENT_CRUX_MIN_SCORE",
    "PersistentCrux",
    "build_arbitration",
    "build_reversal",
    "crux_arbitration_enabled",
    "CoherenceIssue",
    "CoherenceReport",
    "CruxEntry",
    "CruxReceipt",
    "DEFAULT_CRUX_LOAD_BEARING_THRESHOLD",
    "DEFAULT_DELTA_LOSS_THRESHOLD",
    "DecayReason",
    "DecaySignal",
    "FollowupProposal",
    "FragilityReport",
    "OrgTruthMapReport",
    "QuarantineDecision",
    "QuarantinePolicy",
    "RepairSpec",
    "StressPerturbation",
    "StressTestResult",
    "IncoherenceKind",
    "apply_quarantine_policy",
    "build_crux_receipt",
    "build_truth_map",
    "build_truth_map_from_manifests",
    "coherence_monitor_enabled",
    "crux_receipt_enabled",
    "enable_crux_receipt",
    "enable_epistemic_followup",
    "enable_repair_pipeline",
    "epistemic_followup_enabled",
    "evaluate_unit",
    "from_belief_node",
    "propose_followup_for_crux",
    "propose_followup_for_cruxset",
    "propose_followup_for_failed_claim",
    "propose_repair",
    "quarantine_policy_enabled",
    "repair_pipeline_enabled",
    "run_stress_test",
    "scan_coherence",
    "stress_test_enabled",
]


def epistemic_followup_enabled() -> bool:
    """Return True if callers should act on DIC-17 follow-up proposals.

    Reads ``ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED`` from the process
    environment. Default is False; construction of proposals is
    always safe, but filing them on GitHub must be gated.
    """
    raw = str(os.environ.get("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_epistemic_followup() -> None:
    """Enable the DIC-17 follow-up bridge for the current process."""
    os.environ["ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED"] = "1"
