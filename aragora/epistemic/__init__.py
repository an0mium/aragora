"""Epistemic CI and crux-engine helpers (DIC-13..22 tranche).

Exposes:
- DIC-14: executable claim verification (:class:`ClaimVerifier`)
- DIC-17: follow-up-issue bridge for load-bearing cruxes and
  sharply-losing claims (:class:`FollowupProposal`,
  :func:`propose_followup_for_crux`, :func:`propose_followup_for_cruxset`,
  :func:`propose_followup_for_failed_claim`)
- DIC-20: epistemic decay monitor (:class:`DecaySignal`,
  :class:`DecayReason`, :func:`evaluate_unit`)
- DIC-26: belief coherence monitor (:class:`BeliefEntry`,
  :class:`CoherenceReport`, :func:`scan_coherence`)

See ``docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md`` for the full
DIC-13..22 sequence and ``docs/status/NEXT_STEPS_CANONICAL.md`` for
the queue-governance activation gate.
"""

from __future__ import annotations

import os

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
from .decay_monitor import DecayReason, DecaySignal, evaluate_unit
from .followup import (
    DEFAULT_CRUX_LOAD_BEARING_THRESHOLD,
    DEFAULT_DELTA_LOSS_THRESHOLD,
    FollowupProposal,
    propose_followup_for_crux,
    propose_followup_for_cruxset,
    propose_followup_for_failed_claim,
)

__all__ = [
    "BeliefEntry",
    "ClaimResult",
    "ClaimStatus",
    "ClaimVerifier",
    "CoherenceIssue",
    "CoherenceReport",
    "DEFAULT_CRUX_LOAD_BEARING_THRESHOLD",
    "DEFAULT_DELTA_LOSS_THRESHOLD",
    "DecayReason",
    "DecaySignal",
    "FollowupProposal",
    "IncoherenceKind",
    "coherence_monitor_enabled",
    "enable_epistemic_followup",
    "epistemic_followup_enabled",
    "evaluate_unit",
    "from_belief_node",
    "propose_followup_for_crux",
    "propose_followup_for_cruxset",
    "propose_followup_for_failed_claim",
    "scan_coherence",
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
