"""Tests for :mod:`aragora.pdb.budget`.

Covers every outcome of :func:`evaluate_budget`:

- ``ALLOWED`` — full roster fits in both caps
- ``BUDGET_DEGRADED`` — optional slots dropped greedily to fit
- ``BUDGET_EXCEEDED`` — minimum safe roster cannot be funded under
  either the per-brief or the per-day cap

Plus reservation + ledger invariants:

- reserving debits the ledger
- releasing unused credits the ledger
- charging after release is a hard error
- double-release is a no-op
- :func:`reserve` refuses to reserve against a BUDGET_EXCEEDED decision
- per-day cap is enforced separately from per-brief cap
- :class:`ReviewBudget` can tighten but not loosen the per-brief cap
- no silent downgrade to ``metadata_heuristic`` — the caller gets an
  explicit denial
"""

from __future__ import annotations

import pytest

from aragora.pdb.budget import (
    DEFAULT_SLOT_COST_USD,
    DEFAULT_SYNTHESIS_COST_USD,
    PDBBudgetDecision,
    PDBBudgetLedger,
    PDBBudgetReservation,
    PDBBudgetStatus,
    SlotCostEstimator,
    estimate_slot_costs,
    evaluate_budget,
    per_brief_cap_usd,
    reserve,
)
from aragora.pdb.panel_config import PDBBudgetConfig, PDBPanelSlot
from aragora.review.policy import ReviewBudget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _slot(slot_id: str, family: str, required: bool = False, lens: str = "core") -> PDBPanelSlot:
    return PDBPanelSlot(
        slot_id=slot_id,
        review_role="logic_reviewer" if lens == "core" else "skeptic",
        lens=lens,
        family=family,
        candidates=(f"{family}-cli",),
        required=required,
    )


@pytest.fixture
def budget_cfg() -> PDBBudgetConfig:
    # 8.00 per brief, 200.00 per day, per spec default
    return PDBBudgetConfig(
        per_brief_usd=8.0, per_day_usd=200.0, reserve_for_manual_escalation_usd=40.0
    )


@pytest.fixture
def ledger(budget_cfg: PDBBudgetConfig) -> PDBBudgetLedger:
    return PDBBudgetLedger(daily_cap_usd=budget_cfg.per_day_usd)


@pytest.fixture
def slots() -> tuple[PDBPanelSlot, ...]:
    return (
        _slot("claude_core", "claude", required=True, lens="core"),
        _slot("gpt_core", "gpt", required=True, lens="core"),
        _slot("gemini_h", "gemini", lens="heterodox"),
        _slot("grok_h", "grok", lens="heterodox"),
        _slot("deepseek_h", "deepseek", lens="heterodox"),
        _slot("mistral_r", "mistral", lens="regulatory"),
    )


@pytest.fixture
def synthesizer(slots: tuple[PDBPanelSlot, ...]) -> PDBPanelSlot:
    return slots[0]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def test_default_estimator_is_flat(slots: tuple[PDBPanelSlot, ...]) -> None:
    est = estimate_slot_costs(slots)
    assert all(cost == DEFAULT_SLOT_COST_USD for cost in est.values())
    assert list(est.keys()) == [s.slot_id for s in slots]


def test_custom_estimator_is_honored(slots: tuple[PDBPanelSlot, ...]) -> None:
    class ExpensiveCore(SlotCostEstimator):
        def estimate(self, slot: PDBPanelSlot) -> float:
            return 3.0 if slot.lens == "core" else 0.25

    est = estimate_slot_costs(slots, ExpensiveCore())
    assert est["claude_core"] == 3.0
    assert est["gemini_h"] == 0.25


# ---------------------------------------------------------------------------
# evaluate_budget — ALLOWED
# ---------------------------------------------------------------------------


def test_evaluate_budget_allows_full_roster_when_generous(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    # Cheap slots + cheap synthesis → full roster fits in $8.00 cap.
    estimator = type(
        "Tiny",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 0.5},
    )()
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=1.0,
    )
    assert decision.status is PDBBudgetStatus.ALLOWED
    assert set(decision.funded_slots) == {s.slot_id for s in slots}
    assert decision.dropped_slots == ()
    assert decision.total_estimated_usd == pytest.approx(0.5 * len(slots) + 1.0)


# ---------------------------------------------------------------------------
# evaluate_budget — BUDGET_DEGRADED
# ---------------------------------------------------------------------------


def test_evaluate_budget_degrades_optional_slots_greedy(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    # 6 slots × $1.50 + $1.50 synthesis = $10.50 — overruns the $8.00 cap.
    estimator = type(
        "Mid",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 1.5},
    )()
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=1.5,
    )
    assert decision.status is PDBBudgetStatus.BUDGET_DEGRADED
    # Both required core slots must survive the degrade
    assert "claude_core" in decision.funded_slots
    assert "gpt_core" in decision.funded_slots
    # Some optional slots must have been dropped
    assert decision.dropped_slots
    # Every dropped slot must be optional (required=False)
    dropped_by_id = {s.slot_id for s in slots if not s.required}
    assert set(decision.dropped_slots) <= dropped_by_id
    # Total estimated is <= per_brief_usd cap (within rounding)
    assert decision.total_estimated_usd <= budget_cfg.per_brief_usd + 1e-6
    assert decision.binding_cap == "per_brief_usd"


def test_evaluate_budget_degrades_preserves_panel_order(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    estimator = type(
        "Mid",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 1.5},
    )()
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=1.5,
    )
    # funded_slots preserves the panel's original ordering
    panel_order = [s.slot_id for s in slots]
    assert decision.funded_slots == tuple(
        sid for sid in panel_order if sid in set(decision.funded_slots)
    )
    assert decision.dropped_slots == tuple(
        sid for sid in panel_order if sid in set(decision.dropped_slots)
    )


# ---------------------------------------------------------------------------
# evaluate_budget — BUDGET_EXCEEDED
# ---------------------------------------------------------------------------


def test_evaluate_budget_rejects_when_core_plus_synth_overruns_per_brief(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    # Two core slots at $10 + synth $5 = $25 > $8 cap.
    estimator = type(
        "Huge",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 10.0},
    )()
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=5.0,
    )
    assert decision.status is PDBBudgetStatus.BUDGET_EXCEEDED
    assert decision.funded_slots == ()
    assert decision.binding_cap == "per_brief_usd"
    assert "per_brief_usd" in decision.reason or "per_brief" in decision.reason


def test_evaluate_budget_rejects_when_daily_pool_exhausted(
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    budget = PDBBudgetConfig(
        per_brief_usd=50.0,
        per_day_usd=60.0,
        reserve_for_manual_escalation_usd=0.0,
    )
    ledger = PDBBudgetLedger(daily_cap_usd=budget.per_day_usd, spent_today_usd=58.0)
    estimator = type(
        "Mid",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 1.5},
    )()
    # Minimum roster: claude_core + gpt_core + synth = $1.5 + $1.5 + $1.5 = $4.5
    # Daily headroom: $60 - $58 = $2 — cannot fit minimum.
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=1.5,
    )
    assert decision.status is PDBBudgetStatus.BUDGET_EXCEEDED
    assert decision.binding_cap == "per_day_usd"


def test_no_silent_downgrade_on_budget_exceeded(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    estimator = type(
        "Huge",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 20.0},
    )()
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=5.0,
    )
    # The spec is categorical: budget_exceeded MUST be explicit, not a
    # silent fallback to a cheaper roster.
    assert decision.status is PDBBudgetStatus.BUDGET_EXCEEDED
    assert decision.funded_slots == ()
    # Reservation must refuse the decision as well.
    with pytest.raises(RuntimeError):
        reserve(decision, ledger)
    # Ledger must not have been debited.
    assert ledger.spent_today_usd == 0.0


# ---------------------------------------------------------------------------
# Reservation lifecycle
# ---------------------------------------------------------------------------


def test_reserve_debits_ledger_and_release_returns_unused(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    estimator = type(
        "Tiny",
        (SlotCostEstimator,),
        {"estimate": lambda self, slot: 0.5},
    )()
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=1.0,
    )
    assert decision.status is PDBBudgetStatus.ALLOWED
    reservation = reserve(decision, ledger)
    assert ledger.spent_today_usd == pytest.approx(decision.total_estimated_usd)
    assert isinstance(reservation, PDBBudgetReservation)

    # Charge partial real spend.
    reservation.charge(0.5)
    reservation.charge(1.0)
    assert reservation.actual_spend_usd == pytest.approx(1.5)
    unused = reservation.released_unused_usd()
    assert unused == pytest.approx(decision.total_estimated_usd - 1.5)

    released = reservation.release_unused()
    assert released == pytest.approx(unused)
    # Ledger must reflect the actual spend after release
    assert ledger.spent_today_usd == pytest.approx(1.5)


def test_release_unused_idempotent(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=SlotCostEstimator(default_usd=0.5),
        synthesis_cost_usd=1.0,
    )
    reservation = reserve(decision, ledger)
    reservation.release_unused()
    assert reservation.release_unused() == 0.0


def test_charge_after_release_is_rejected(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=SlotCostEstimator(default_usd=0.5),
        synthesis_cost_usd=1.0,
    )
    reservation = reserve(decision, ledger)
    reservation.release_unused()
    with pytest.raises(RuntimeError):
        reservation.charge(0.1)


def test_charge_negative_rejected(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=SlotCostEstimator(default_usd=0.5),
        synthesis_cost_usd=1.0,
    )
    reservation = reserve(decision, ledger)
    with pytest.raises(ValueError):
        reservation.charge(-0.1)


# ---------------------------------------------------------------------------
# per_brief_cap_usd
# ---------------------------------------------------------------------------


def test_review_budget_tightens_per_brief_cap(budget_cfg: PDBBudgetConfig) -> None:
    review = ReviewBudget(per_pr_usd_cap=3.0)
    assert per_brief_cap_usd(budget_cfg, review) == pytest.approx(3.0)


def test_review_budget_cannot_loosen_per_brief_cap(budget_cfg: PDBBudgetConfig) -> None:
    review = ReviewBudget(per_pr_usd_cap=100.0)
    # PDB default ($8) is stricter and wins.
    assert per_brief_cap_usd(budget_cfg, review) == pytest.approx(budget_cfg.per_brief_usd)


def test_review_budget_zero_means_no_override(budget_cfg: PDBBudgetConfig) -> None:
    review = ReviewBudget(per_pr_usd_cap=0.0)
    assert per_brief_cap_usd(budget_cfg, review) == pytest.approx(budget_cfg.per_brief_usd)


# ---------------------------------------------------------------------------
# Ledger bookkeeping
# ---------------------------------------------------------------------------


def test_ledger_debit_and_credit_round_trip() -> None:
    ledger = PDBBudgetLedger(daily_cap_usd=100.0)
    ledger.debit(30.0)
    ledger.credit(5.0)
    assert ledger.spent_today_usd == pytest.approx(25.0)
    assert ledger.headroom_usd() == pytest.approx(75.0)


def test_ledger_credit_cannot_go_negative() -> None:
    ledger = PDBBudgetLedger(daily_cap_usd=100.0)
    ledger.debit(5.0)
    ledger.credit(100.0)  # over-credit
    assert ledger.spent_today_usd == 0.0


def test_decision_to_dict_is_json_friendly(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=SlotCostEstimator(default_usd=0.5),
        synthesis_cost_usd=1.0,
    )
    import json

    payload = decision.to_dict()
    # Round-trip through JSON so we know every value is serializable.
    assert json.loads(json.dumps(payload))["status"] == decision.status.value


def test_decision_explicit_budget_exceeded_status_value(
    budget_cfg: PDBBudgetConfig,
    ledger: PDBBudgetLedger,
    slots: tuple[PDBPanelSlot, ...],
    synthesizer: PDBPanelSlot,
) -> None:
    estimator = SlotCostEstimator(default_usd=20.0)
    decision = evaluate_budget(
        findings_slots=slots,
        synthesizer_slot=synthesizer,
        budget=budget_cfg,
        ledger=ledger,
        estimator=estimator,
        synthesis_cost_usd=5.0,
    )
    # No silent rewrite to "metadata_heuristic" — the caller sees exactly
    # "budget_exceeded" so PR3 transport can honor the contract.
    assert decision.status.value == "budget_exceeded"
    assert decision.to_dict()["status"] == "budget_exceeded"
