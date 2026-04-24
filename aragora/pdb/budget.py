"""Back-compat shim for :mod:`aragora.brief_engine.budget`.

The generic Protocol B budget layer moved to
:mod:`aragora.brief_engine.budget` in the Phase 1 brief-engine
extraction. This module re-exports the same public surface under both
the new ``Brief*`` names and the legacy ``PDB*`` aliases so existing
Mode 3 callers continue to work unchanged.

Do not add new budget logic here — put it in
:mod:`aragora.brief_engine.budget` instead.
"""

from __future__ import annotations

from aragora.brief_engine.budget import (
    DEFAULT_SLOT_COST_USD,
    DEFAULT_SYNTHESIS_COST_USD,
    BriefBudgetDecision,
    BriefBudgetLedger,
    BriefBudgetReservation,
    BriefBudgetStatus,
    SlotCostEstimator,
    estimate_slot_costs,
    evaluate_budget,
    per_brief_cap_usd,
    reserve,
)

# Legacy PDB-named aliases.
PDBBudgetDecision = BriefBudgetDecision
PDBBudgetLedger = BriefBudgetLedger
PDBBudgetReservation = BriefBudgetReservation
PDBBudgetStatus = BriefBudgetStatus

__all__ = [
    "DEFAULT_SLOT_COST_USD",
    "DEFAULT_SYNTHESIS_COST_USD",
    "PDBBudgetDecision",
    "PDBBudgetLedger",
    "PDBBudgetReservation",
    "PDBBudgetStatus",
    "SlotCostEstimator",
    "estimate_slot_costs",
    "evaluate_budget",
    "per_brief_cap_usd",
    "reserve",
]
