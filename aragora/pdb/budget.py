"""PDB Mode 3 Protocol B budget reservation and denial logic.

Layered on top of the :class:`aragora.review.policy.ReviewPolicy` +
:class:`aragora.review.policy.ReviewBudget` schema, this module adds
Protocol-B-specific reservation semantics:

- estimate the full configured-panel spend before any findings round
- reserve spend against per-brief and per-day caps
- release unused reserve once a run finishes (or never starts)
- record which slots the reserved budget actually funded

**Never silently degrade to metadata-heuristic output.** A breach
yields an explicit :data:`PDBBudgetStatus.BUDGET_EXCEEDED` decision
that the executor propagates to its caller. Callers may pair the
denial with a pre-existing heuristic packet for UX, but this module
does not fabricate one.

The module is deterministic and side-effect free: a
:class:`PDBBudgetLedger` captures the rolling daily spend the caller
is responsible for maintaining (the storage layer in PR1 is one
candidate store; callers may also supply an in-memory ledger for
tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from aragora.pdb.panel_config import PDBBudgetConfig, PDBPanelSlot
from aragora.review.policy import ReviewBudget

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
]


DEFAULT_SLOT_COST_USD = 0.85
"""Default per-slot findings+critique cost assumption (USD).

Matches the heuristic upper band surfaced by
:class:`aragora.swarm.pr_review_protocol.PRReviewProtocol.build_packet`
so PDB estimates don't diverge from the metadata-heuristic packet the
UI already shows.
"""

DEFAULT_SYNTHESIS_COST_USD = 1.1
"""Default cost reservation for the single synthesis pass (USD)."""


# ---------------------------------------------------------------------------
# Status / decision / reservation dataclasses
# ---------------------------------------------------------------------------


class PDBBudgetStatus(str, Enum):
    """Outcome of :func:`evaluate_budget`.

    - ``ALLOWED`` — the full configured roster fits inside both caps.
    - ``BUDGET_DEGRADED`` — at least one optional slot was dropped to
      fit inside the per-brief cap while preserving the minimum safe
      roster (both core slots + synthesis). Execution may proceed.
    - ``BUDGET_EXCEEDED`` — neither the full nor the minimum safe
      roster fits. Execution MUST NOT proceed; the caller is expected
      to surface the denial and fall back to the pre-existing
      heuristic packet labeled as such.
    """

    ALLOWED = "allowed"
    BUDGET_DEGRADED = "budget_degraded"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True, slots=True)
class PDBBudgetDecision:
    """The budget verdict plus enough context to explain itself."""

    status: PDBBudgetStatus
    total_estimated_usd: float
    per_brief_cap_usd: float
    per_day_cap_usd: float
    per_day_spent_before_usd: float
    per_day_remaining_before_usd: float
    funded_slots: tuple[str, ...]
    dropped_slots: tuple[str, ...]
    slot_estimates_usd: Mapping[str, float]
    synthesis_estimate_usd: float
    reason: str
    binding_cap: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "total_estimated_usd": round(self.total_estimated_usd, 4),
            "per_brief_cap_usd": round(self.per_brief_cap_usd, 4),
            "per_day_cap_usd": round(self.per_day_cap_usd, 4),
            "per_day_spent_before_usd": round(self.per_day_spent_before_usd, 4),
            "per_day_remaining_before_usd": round(self.per_day_remaining_before_usd, 4),
            "funded_slots": list(self.funded_slots),
            "dropped_slots": list(self.dropped_slots),
            "slot_estimates_usd": {k: round(v, 4) for k, v in self.slot_estimates_usd.items()},
            "synthesis_estimate_usd": round(self.synthesis_estimate_usd, 4),
            "reason": self.reason,
            "binding_cap": self.binding_cap,
        }


@dataclass(slots=True)
class PDBBudgetReservation:
    """Mutable reservation handle for a single Protocol B run.

    The executor reserves once up-front (``reserve_total_usd``), then
    charges actual spend as each phase completes via
    :meth:`charge`. When findings/critique finish, any over-reserve
    that did not land is released via :meth:`release_unused` so the
    rolling daily pool recovers headroom.

    ``release_unused`` is idempotent: re-releasing a finalized
    reservation is a no-op. Charging after finalization raises
    :class:`RuntimeError` because the budget ledger would silently
    under-report daily spend otherwise.
    """

    decision: PDBBudgetDecision
    reserve_total_usd: float
    actual_spend_usd: float = 0.0
    finalized: bool = False
    _ledger: "PDBBudgetLedger | None" = field(default=None, repr=False)

    def charge(self, usd: float) -> None:
        """Record actual spend for an executed slot or synthesis pass.

        Raises :class:`RuntimeError` after :meth:`release_unused` has
        been called to prevent silent under-reporting of daily spend.
        """
        if self.finalized:
            raise RuntimeError(
                "cannot charge against a finalized PDB budget reservation; "
                "the ledger has already been reconciled"
            )
        if usd < 0:
            raise ValueError(f"charge must be >= 0; got {usd}")
        self.actual_spend_usd = round(self.actual_spend_usd + usd, 6)

    def released_unused_usd(self) -> float:
        """Return the amount that will be returned to the daily pool."""
        return max(0.0, round(self.reserve_total_usd - self.actual_spend_usd, 6))

    def release_unused(self) -> float:
        """Finalize the reservation and return unused spend to the ledger."""
        if self.finalized:
            return 0.0
        released = self.released_unused_usd()
        self.finalized = True
        if self._ledger is not None:
            # The ledger was debited ``reserve_total_usd`` at reserve-time.
            # Return the unused portion so cumulative-daily-spend reflects
            # the actual (not reserved) outlay.
            self._ledger.credit(released)
        return released


@dataclass(slots=True)
class PDBBudgetLedger:
    """Simple in-process rolling-daily-spend ledger.

    Callers that want to persist daily spend across workers should
    subclass or compose this via dependency injection. The in-memory
    default is sufficient for PR2's mocked-provider tests; PR3 wires a
    durable backing store (storage from PR1, Redis, or similar).
    """

    daily_cap_usd: float
    spent_today_usd: float = 0.0

    def headroom_usd(self) -> float:
        return max(0.0, round(self.daily_cap_usd - self.spent_today_usd, 6))

    def debit(self, usd: float) -> None:
        if usd < 0:
            raise ValueError(f"debit must be >= 0; got {usd}")
        self.spent_today_usd = round(self.spent_today_usd + usd, 6)

    def credit(self, usd: float) -> None:
        if usd < 0:
            raise ValueError(f"credit must be >= 0; got {usd}")
        self.spent_today_usd = round(max(0.0, self.spent_today_usd - usd), 6)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class SlotCostEstimator:
    """Per-slot cost-estimation strategy.

    Default implementation returns :data:`DEFAULT_SLOT_COST_USD` for
    every slot. Tests and PR3 callers can inject bespoke estimators
    (e.g., one that consults a model-cost table). The executor only
    calls :meth:`estimate`; subclassing is NOT required.
    """

    def __init__(self, default_usd: float = DEFAULT_SLOT_COST_USD) -> None:
        self._default_usd = default_usd

    def estimate(self, slot: PDBPanelSlot) -> float:
        return self._default_usd


def estimate_slot_costs(
    slots: Sequence[PDBPanelSlot],
    estimator: SlotCostEstimator | None = None,
) -> dict[str, float]:
    """Return an ordered mapping ``slot_id -> estimated cost`` in USD."""
    est = estimator or SlotCostEstimator()
    return {slot.slot_id: est.estimate(slot) for slot in slots}


def per_brief_cap_usd(budget: PDBBudgetConfig, review_budget: ReviewBudget | None = None) -> float:
    """Resolve the effective per-brief cap.

    When the caller supplies a :class:`ReviewBudget`, the stricter of
    ``ReviewBudget.per_pr_usd_cap`` and
    :attr:`PDBBudgetConfig.per_brief_usd` wins. This mirrors the
    "compose with, do not replace" posture of
    :mod:`aragora.review.policy`: the repo-level review budget can
    tighten the PDB default but not silently loosen it.
    """
    cap = budget.per_brief_usd
    if review_budget is not None and review_budget.per_pr_usd_cap > 0:
        cap = min(cap, review_budget.per_pr_usd_cap)
    return float(cap)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_budget(
    *,
    findings_slots: Sequence[PDBPanelSlot],
    synthesizer_slot: PDBPanelSlot,
    budget: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    estimator: SlotCostEstimator | None = None,
    synthesis_cost_usd: float = DEFAULT_SYNTHESIS_COST_USD,
    review_budget: ReviewBudget | None = None,
) -> PDBBudgetDecision:
    """Evaluate a panel roster against the per-brief and per-day caps.

    Returns a :class:`PDBBudgetDecision` whose ``status`` is one of
    :data:`PDBBudgetStatus.ALLOWED`, :data:`PDBBudgetStatus.BUDGET_DEGRADED`,
    or :data:`PDBBudgetStatus.BUDGET_EXCEEDED`. The caller is expected
    to then reserve the decided total via :func:`reserve` (or directly
    construct a :class:`PDBBudgetReservation`) before running findings.

    Rules (per ``docs/plans/2026-04-21-pdb-mode3-pr2-spec.md`` §Budget
    rules):

    1. If both required core slots + one synthesis pass fit in the
       per-brief cap AND the per-day headroom, execution may run at
       full roster → ``ALLOWED``.
    2. If optional non-core slots don't fit, drop them greedily from
       the end of ``findings_slots`` while preserving all required
       slots plus the synthesizer → ``BUDGET_DEGRADED``.
    3. If required slots + synthesis don't fit in either cap → return
       ``BUDGET_EXCEEDED`` with no silent fallback.
    """
    if not findings_slots:
        raise ValueError("findings_slots must be non-empty")
    if synthesis_cost_usd < 0:
        raise ValueError(f"synthesis_cost_usd must be >= 0; got {synthesis_cost_usd}")

    per_brief = per_brief_cap_usd(budget, review_budget)
    per_day_spent_before = ledger.spent_today_usd
    per_day_remaining_before = ledger.headroom_usd()

    slot_estimates = estimate_slot_costs(findings_slots, estimator)

    required_slot_ids = {slot.slot_id for slot in findings_slots if slot.required}
    optional_slot_ids = [slot.slot_id for slot in findings_slots if not slot.required]

    # Minimum safe roster is all required slots plus the synthesizer.
    minimum_slot_ids = set(required_slot_ids) | {synthesizer_slot.slot_id}
    minimum_cost = (
        sum(slot_estimates.get(sid, 0.0) for sid in minimum_slot_ids) + synthesis_cost_usd
    )

    # If the floor can't fit under either cap, fail closed.
    if minimum_cost > per_brief or minimum_cost > per_day_remaining_before:
        binding = "per_brief_usd" if minimum_cost > per_brief else "per_day_usd"
        return PDBBudgetDecision(
            status=PDBBudgetStatus.BUDGET_EXCEEDED,
            total_estimated_usd=minimum_cost,
            per_brief_cap_usd=per_brief,
            per_day_cap_usd=budget.per_day_usd,
            per_day_spent_before_usd=per_day_spent_before,
            per_day_remaining_before_usd=per_day_remaining_before,
            funded_slots=(),
            dropped_slots=tuple(slot.slot_id for slot in findings_slots),
            slot_estimates_usd=slot_estimates,
            synthesis_estimate_usd=synthesis_cost_usd,
            reason=(
                f"minimum safe roster costs ${minimum_cost:.2f} which exceeds "
                f"{binding} (per_brief cap=${per_brief:.2f}, "
                f"per_day remaining=${per_day_remaining_before:.2f})"
            ),
            binding_cap=binding,
        )

    # Greedy fit: keep all required slots in their configured order,
    # then pile on optional slots until adding one would overflow
    # either cap.
    ordered_required: list[str] = [
        sid for sid in (s.slot_id for s in findings_slots) if sid in required_slot_ids
    ]
    funded: list[str] = list(ordered_required)
    running_total = sum(slot_estimates[sid] for sid in funded) + synthesis_cost_usd

    cap_headroom = min(per_brief, per_day_remaining_before)
    dropped: list[str] = []
    for sid in optional_slot_ids:
        cost = slot_estimates[sid]
        if running_total + cost <= cap_headroom + 1e-9:
            funded.append(sid)
            running_total += cost
        else:
            dropped.append(sid)

    status = PDBBudgetStatus.ALLOWED if not dropped else PDBBudgetStatus.BUDGET_DEGRADED
    if dropped:
        binding = "per_brief_usd" if per_brief <= per_day_remaining_before else "per_day_usd"
        reason = (
            f"dropped optional slots {dropped} to fit within {binding} "
            f"(cap_headroom=${cap_headroom:.2f}, ran=${running_total:.2f})"
        )
        binding_cap = binding
    else:
        reason = (
            f"full configured roster funded at ${running_total:.2f} "
            f"within cap_headroom=${cap_headroom:.2f}"
        )
        binding_cap = None

    # Preserve panel order in funded_slots so the executor iterates
    # findings in the intended order.
    panel_order = [slot.slot_id for slot in findings_slots]
    funded_ordered = tuple(sid for sid in panel_order if sid in set(funded))
    dropped_ordered = tuple(sid for sid in panel_order if sid not in set(funded))

    return PDBBudgetDecision(
        status=status,
        total_estimated_usd=running_total,
        per_brief_cap_usd=per_brief,
        per_day_cap_usd=budget.per_day_usd,
        per_day_spent_before_usd=per_day_spent_before,
        per_day_remaining_before_usd=per_day_remaining_before,
        funded_slots=funded_ordered,
        dropped_slots=dropped_ordered,
        slot_estimates_usd=slot_estimates,
        synthesis_estimate_usd=synthesis_cost_usd,
        reason=reason,
        binding_cap=binding_cap,
    )


def reserve(
    decision: PDBBudgetDecision,
    ledger: PDBBudgetLedger,
) -> PDBBudgetReservation:
    """Materialize a reservation and debit the rolling daily ledger.

    Raises :class:`RuntimeError` if called with a
    :data:`PDBBudgetStatus.BUDGET_EXCEEDED` decision — such a decision
    must not be converted into a reservation, because that would bypass
    the explicit-denial contract.
    """
    if decision.status is PDBBudgetStatus.BUDGET_EXCEEDED:
        raise RuntimeError(
            "cannot reserve against a BUDGET_EXCEEDED decision; "
            "executor must fail closed with an explicit denial"
        )
    ledger.debit(decision.total_estimated_usd)
    return PDBBudgetReservation(
        decision=decision,
        reserve_total_usd=decision.total_estimated_usd,
        _ledger=ledger,
    )
