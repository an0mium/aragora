"""Tests for DIC-28 Proactive Crux Gardening (aragora/epistemic/gardening.py).

No network, queue, or database access.  All inputs are synthetic.

Acceptance criteria (from issue #6222):
  (a) resolved crux whose evidence went stale is surfaced
  (b) outstanding crux with reduced fragility (below threshold) is healthy
  (c) new contradiction emerging on a crux family is flagged
"""

from __future__ import annotations

import os

import pytest

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus
from aragora.epistemic.coherence import BeliefEntry, CoherenceIssue, IncoherenceKind
from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt
from aragora.epistemic.gardening import (
    DEFAULT_FRAGILITY_SHIFT_THRESHOLD,
    CruxGardeningResult,
    GardeningReport,
    crux_gardening_enabled,
    enable_crux_gardening,
    garden_outstanding_crux,
    garden_resolved_crux,
    run_gardening_pass,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(
    crux_id: str = "crux-1",
    affected_claims: list[str] | None = None,
) -> CruxEntry:
    return CruxEntry(
        crux_id=crux_id,
        statement="Should we expand B2 guard now?",
        load_bearing_score=0.82,
        uncertainty_score=0.4,
        contesting_agents=["claude", "codex"],
        affected_claims=affected_claims or ["claim-a", "claim-b"],
        resolution_impact=0.9,
    )


def _receipt(crux_id: str = "crux-1", affected_claims: list[str] | None = None) -> CruxReceipt:
    entry = _entry(crux_id=crux_id, affected_claims=affected_claims)
    return CruxReceipt(
        receipt_id="rcpt-001",
        debate_id="debate-001",
        question="expand B2?",
        cruxes=[entry],
        convergence_barrier=0.75,
        counterfactuals=[],
        agents=["claude", "codex"],
        rounds=3,
        metadata={},
        checksum="a" * 64,
    )


def _stale_result(claim_id: str) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        status=ClaimStatus.STALE,
        message="evidence stale",
        severity="warning",
        allowed_action="report_only",
    )


def _pass_result(claim_id: str) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        status=ClaimStatus.PASS,
        message="ok",
        severity="info",
        allowed_action="report_only",
    )


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------


def test_disabled_by_default() -> None:
    os.environ.pop("ARAGORA_CRUX_GARDENING_ENABLED", None)
    assert crux_gardening_enabled() is False


def test_enable_sets_env() -> None:
    os.environ.pop("ARAGORA_CRUX_GARDENING_ENABLED", None)
    enable_crux_gardening()
    assert crux_gardening_enabled() is True
    os.environ.pop("ARAGORA_CRUX_GARDENING_ENABLED", None)


def test_override_kwarg() -> None:
    assert crux_gardening_enabled(override=True) is True
    assert crux_gardening_enabled(override=False) is False


# ---------------------------------------------------------------------------
# (a) Resolved crux — stale evidence surfaces
# ---------------------------------------------------------------------------


def test_resolved_crux_stale_evidence_surfaced() -> None:
    receipt = _receipt(affected_claims=["claim-a"])
    claim_results = {"claim-a": _stale_result("claim-a")}
    results = garden_resolved_crux(receipt, claim_results=claim_results)
    assert len(results) == 1
    r = results[0]
    assert r.status == "stale_evidence"
    assert "claim-a" in r.detail


def test_resolved_crux_healthy_when_all_pass() -> None:
    receipt = _receipt(affected_claims=["claim-a", "claim-b"])
    claim_results = {
        "claim-a": _pass_result("claim-a"),
        "claim-b": _pass_result("claim-b"),
    }
    results = garden_resolved_crux(receipt, claim_results=claim_results)
    assert results[0].status == "healthy"


# ---------------------------------------------------------------------------
# (c) New contradiction on crux family is flagged
# ---------------------------------------------------------------------------


def test_coherence_issue_referencing_only_crux_id_is_not_matched() -> None:
    """belief_ids that match only the crux_id (not a claim) must not surface."""
    receipt = _receipt(crux_id="crux-1", affected_claims=["claim-a"])
    contradiction = CoherenceIssue(
        kind=IncoherenceKind.CONTRADICTION,
        belief_ids=("crux-1",),  # crux_id only — different namespace from claim IDs
        detail="crux-id-only reference",
        severity="warning",
    )
    results = garden_resolved_crux(
        receipt,
        claim_results={"claim-a": _pass_result("claim-a")},
        coherence_issues=[contradiction],
    )
    assert results[0].status == "healthy"


def test_resolved_crux_new_contradiction_flagged() -> None:
    receipt = _receipt(affected_claims=["claim-a"])
    belief = BeliefEntry(belief_id="claim-a", subject="B2 guard", confidence=0.6)
    contradiction = CoherenceIssue(
        kind=IncoherenceKind.CONTRADICTION,
        belief_ids=("claim-a", "claim-x"),
        detail="contradicts claim-x",
        severity="error",
    )
    results = garden_resolved_crux(
        receipt,
        claim_results={"claim-a": _pass_result("claim-a")},
        coherence_issues=[contradiction],
    )
    assert results[0].status == "new_contradiction"
    assert "contradiction" in results[0].detail


def test_evidence_conflict_does_not_override_stale() -> None:
    """Stale evidence takes priority over coherence issues."""
    receipt = _receipt(affected_claims=["claim-a"])
    contradiction = CoherenceIssue(
        kind=IncoherenceKind.CONTRADICTION,
        belief_ids=("claim-a",),
        detail="conflict",
        severity="warning",
    )
    results = garden_resolved_crux(
        receipt,
        claim_results={"claim-a": _stale_result("claim-a")},
        coherence_issues=[contradiction],
    )
    assert results[0].status == "stale_evidence"


# ---------------------------------------------------------------------------
# (b) Outstanding crux — reduced fragility stays healthy
# ---------------------------------------------------------------------------


def test_outstanding_crux_fragility_decrease_below_threshold_is_healthy() -> None:
    entry = _entry()
    result = garden_outstanding_crux(
        entry,
        previous_fragility=0.5,
        current_fragility=0.45,  # delta=0.05 < 0.15
    )
    assert result.status == "healthy"


def test_outstanding_crux_fragility_increase_above_threshold_surfaces() -> None:
    entry = _entry()
    result = garden_outstanding_crux(
        entry,
        previous_fragility=0.3,
        current_fragility=0.6,  # delta=0.3 >= 0.15
    )
    assert result.status == "fragility_shift"
    assert result.previous_fragility == pytest.approx(0.3)
    assert result.current_fragility == pytest.approx(0.6)


def test_outstanding_crux_no_baseline_is_healthy() -> None:
    result = garden_outstanding_crux(_entry(), previous_fragility=None, current_fragility=0.5)
    assert result.status == "healthy"


def test_custom_fragility_threshold() -> None:
    entry = _entry()
    result = garden_outstanding_crux(
        entry,
        previous_fragility=0.5,
        current_fragility=0.6,  # delta=0.1
        fragility_shift_threshold=0.05,  # custom tight threshold
    )
    assert result.status == "fragility_shift"


# ---------------------------------------------------------------------------
# run_gardening_pass summary + to_json round-trip
# ---------------------------------------------------------------------------


def test_run_gardening_pass_summary_counts() -> None:
    resolved = [_receipt(crux_id="r1", affected_claims=["claim-stale"])]
    outstanding = [_entry(crux_id="o1")]
    report = run_gardening_pass(
        resolved,
        outstanding,
        claim_results={"claim-stale": _stale_result("claim-stale")},
        fragility_scores={"o1": (0.3, 0.6)},
    )
    assert isinstance(report, GardeningReport)
    assert report.summary["stale_evidence"] == 1
    assert report.summary["fragility_shift"] == 1
    assert report.summary["healthy"] == 0


def test_run_gardening_pass_to_json_is_deterministic() -> None:
    report = run_gardening_pass([], [_entry()])
    j1 = report.to_json()
    j2 = report.to_json()
    assert j1 == j2
    assert '"schema_version"' in j1


def test_needs_followup_off_by_default() -> None:
    os.environ.pop("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", None)
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        claim_results={"claim-a": _stale_result("claim-a")},
    )
    assert results[0].needs_followup is False


def test_needs_followup_on_when_dic17_gate_open() -> None:
    os.environ["ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED"] = "1"
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        claim_results={"claim-a": _stale_result("claim-a")},
    )
    os.environ.pop("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", None)
    assert results[0].needs_followup is True


def test_needs_followup_via_parameter_bypasses_env() -> None:
    """followup_enabled kwarg injects config without reading env."""
    os.environ.pop("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", None)
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        claim_results={"claim-a": _stale_result("claim-a")},
        followup_enabled=True,
    )
    assert results[0].needs_followup is True


def test_followup_enabled_parameter_on_outstanding_crux() -> None:
    """followup_enabled kwarg works on garden_outstanding_crux too."""
    os.environ.pop("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", None)
    result = garden_outstanding_crux(
        _entry(),
        previous_fragility=0.3,
        current_fragility=0.6,
        followup_enabled=True,
    )
    assert result.needs_followup is True
