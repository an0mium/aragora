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
    STATUS_FRAGILITY_SHIFT,
    STATUS_HEALTHY,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_NEW_CONTRADICTION,
    STATUS_STALE_EVIDENCE,
    CruxGardeningResult,
    GardeningConfig,
    GardeningReport,
    crux_gardening_enabled,
    garden_outstanding_crux,
    garden_resolved_crux,
    run_gardening_pass,
)

# Default falsey config used by direct-call tests; `run_gardening_pass`
# still constructs from env at the boundary.
_CFG = GardeningConfig()


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


def test_from_env_reads_env_flag() -> None:
    os.environ["ARAGORA_CRUX_GARDENING_ENABLED"] = "1"
    assert GardeningConfig.from_env().enabled is True
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
    results = garden_resolved_crux(receipt, config=_CFG, claim_results=claim_results)
    assert len(results) == 1
    r = results[0]
    assert r.status == STATUS_STALE_EVIDENCE
    assert "claim-a" in r.detail


def test_resolved_crux_healthy_when_all_pass() -> None:
    receipt = _receipt(affected_claims=["claim-a", "claim-b"])
    claim_results = {
        "claim-a": _pass_result("claim-a"),
        "claim-b": _pass_result("claim-b"),
    }
    results = garden_resolved_crux(receipt, config=_CFG, claim_results=claim_results)
    assert results[0].status == STATUS_HEALTHY


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
        config=_CFG,
        claim_results={"claim-a": _pass_result("claim-a")},
        coherence_issues=[contradiction],
    )
    assert results[0].status == STATUS_HEALTHY


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
        config=_CFG,
        claim_results={"claim-a": _pass_result("claim-a")},
        coherence_issues=[contradiction],
    )
    assert results[0].status == STATUS_NEW_CONTRADICTION
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
        config=_CFG,
        claim_results={"claim-a": _stale_result("claim-a")},
        coherence_issues=[contradiction],
    )
    assert results[0].status == STATUS_STALE_EVIDENCE


def test_stale_masks_contradiction_in_status_but_preserves_kinds() -> None:
    """Stale priority in status MUST NOT hide the contradiction in coherence_issue_kinds.

    Pins the documented priority policy on ``CruxGardeningResult``: when both
    stale and contradiction signals are present, ``status`` surfaces the
    stale signal, but ``coherence_issue_kinds`` still carries the contradiction
    so downstream consumers can read both.
    """
    receipt = _receipt(affected_claims=["claim-a"])
    contradiction = CoherenceIssue(
        kind=IncoherenceKind.CONTRADICTION,
        belief_ids=("claim-a",),
        detail="conflict",
        severity="error",
    )
    results = garden_resolved_crux(
        receipt,
        config=_CFG,
        claim_results={"claim-a": _stale_result("claim-a")},
        coherence_issues=[contradiction],
    )
    r = results[0]
    assert r.status == STATUS_STALE_EVIDENCE
    assert "contradiction" in r.coherence_issue_kinds


# ---------------------------------------------------------------------------
# (b) Outstanding crux — reduced fragility stays healthy
# ---------------------------------------------------------------------------


def test_outstanding_crux_fragility_decrease_below_threshold_is_healthy() -> None:
    entry = _entry()
    result = garden_outstanding_crux(
        entry,
        config=_CFG,
        previous_fragility=0.5,
        current_fragility=0.45,  # delta=0.05 < 0.15
    )
    assert result.status == STATUS_HEALTHY


def test_outstanding_crux_fragility_increase_above_threshold_surfaces() -> None:
    entry = _entry()
    result = garden_outstanding_crux(
        entry,
        config=_CFG,
        previous_fragility=0.3,
        current_fragility=0.6,  # delta=0.3 >= 0.15
    )
    assert result.status == STATUS_FRAGILITY_SHIFT
    assert result.previous_fragility == pytest.approx(0.3)
    assert result.current_fragility == pytest.approx(0.6)


def test_outstanding_crux_no_baseline_is_insufficient_evidence() -> None:
    """Missing fragility baseline is NOT healthy — it's insufficient evidence."""
    result = garden_outstanding_crux(
        _entry(), config=_CFG, previous_fragility=None, current_fragility=0.5
    )
    assert result.status == STATUS_INSUFFICIENT_EVIDENCE
    assert "cannot evaluate" in result.detail.lower()


def test_custom_fragility_threshold() -> None:
    entry = _entry()
    result = garden_outstanding_crux(
        entry,
        config=_CFG,
        previous_fragility=0.5,
        current_fragility=0.6,  # delta=0.1
        fragility_shift_threshold=0.05,  # custom tight threshold
    )
    assert result.status == STATUS_FRAGILITY_SHIFT


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


def test_needs_followup_off_by_default_config() -> None:
    """Default GardeningConfig() has followup_eligible=False."""
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        config=GardeningConfig(),  # followup_eligible defaults to False
        claim_results={"claim-a": _stale_result("claim-a")},
    )
    assert results[0].needs_followup is False


def test_needs_followup_on_when_config_followup_eligible() -> None:
    """Direct-call sub-functions no longer read env — follow-up is config-driven."""
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        config=GardeningConfig(followup_eligible=True),
        claim_results={"claim-a": _stale_result("claim-a")},
    )
    assert results[0].needs_followup is True


def test_sub_functions_do_not_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """garden_resolved_crux / garden_outstanding_crux must not call os.environ.

    The env is read ONCE at the run_gardening_pass boundary via
    GardeningConfig.from_env(); sub-functions receive the resolved config.
    This test pins that contract by asserting that an env variable set
    between pass-construction and sub-function call does NOT leak into
    the sub-function's behaviour.
    """
    monkeypatch.delenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", raising=False)
    cfg = GardeningConfig(followup_eligible=False)
    # Now set env AFTER cfg is constructed. If the sub-function reads env,
    # the followup flag would flip to True and the test would see
    # needs_followup=True. It must stay False.
    monkeypatch.setenv("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", "1")
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        config=cfg,
        claim_results={"claim-a": _stale_result("claim-a")},
    )
    assert results[0].needs_followup is False


def test_config_threaded_overrides_env() -> None:
    """Explicit GardeningConfig bypasses env-var reads entirely."""
    os.environ.pop("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", None)
    os.environ.pop("ARAGORA_CRUX_GARDENING_ENABLED", None)
    resolved = [_receipt(crux_id="r1", affected_claims=["claim-stale"])]
    report = run_gardening_pass(
        resolved,
        [],
        claim_results={"claim-stale": _stale_result("claim-stale")},
        config=GardeningConfig(enabled=True, followup_eligible=True),
    )
    assert report.summary["stale_evidence"] == 1
    assert report.summary["needs_followup"] == 1


def test_config_followup_on_resolved_crux() -> None:
    """GardeningConfig.followup_eligible=True marks stale crux for followup."""
    receipt = _receipt(affected_claims=["claim-a"])
    results = garden_resolved_crux(
        receipt,
        config=GardeningConfig(followup_eligible=True),
        claim_results={"claim-a": _stale_result("claim-a")},
    )
    assert results[0].needs_followup is True


def test_config_followup_on_outstanding_crux() -> None:
    """GardeningConfig.followup_eligible=True marks fragility-shift crux for followup."""
    result = garden_outstanding_crux(
        _entry(),
        config=GardeningConfig(followup_eligible=True),
        previous_fragility=0.3,
        current_fragility=0.6,
    )
    assert result.needs_followup is True


def test_resolved_crux_insufficient_evidence_when_no_upstream_data() -> None:
    """No claim_results and no coherence_issues → insufficient_evidence, NOT healthy."""
    receipt = _receipt(affected_claims=["claim-a", "claim-b"])
    results = garden_resolved_crux(receipt, config=_CFG)
    assert results[0].status == STATUS_INSUFFICIENT_EVIDENCE
    assert "cannot evaluate" in results[0].detail.lower()


def test_resolved_crux_insufficient_evidence_when_claim_results_dont_cover_crux() -> None:
    """claim_results provided but doesn't contain any of the crux's affected claims."""
    receipt = _receipt(affected_claims=["claim-a", "claim-b"])
    # claim_results has an entry, but for a different claim entirely
    unrelated = {"other-claim": _pass_result("other-claim")}
    results = garden_resolved_crux(receipt, config=_CFG, claim_results=unrelated)
    assert results[0].status == STATUS_INSUFFICIENT_EVIDENCE


def test_resolved_crux_healthy_only_when_evidence_was_observed() -> None:
    """Pins the fix for false-healthy: healthy REQUIRES that we actually looked.

    The pre-repair code returned ``healthy`` when claim_results was missing
    or empty. That's a silent fail-open. Now ``healthy`` means "at least one
    affected claim had a ClaimResult and it passed, AND (if coherence_issues
    were provided) no contradiction touched this crux."
    """
    receipt = _receipt(affected_claims=["claim-a"])
    # Case 1: claim observed + pass + no coherence → healthy
    results_observed = garden_resolved_crux(
        receipt,
        config=_CFG,
        claim_results={"claim-a": _pass_result("claim-a")},
    )
    assert results_observed[0].status == STATUS_HEALTHY

    # Case 2: same crux, nothing observed → insufficient_evidence
    results_unobserved = garden_resolved_crux(receipt, config=_CFG)
    assert results_unobserved[0].status == STATUS_INSUFFICIENT_EVIDENCE


def test_resolved_crux_partial_claim_coverage_is_insufficient_evidence() -> None:
    """One observed claim must not make a multi-claim crux look healthy."""
    receipt = _receipt(affected_claims=["claim-a", "claim-b"])
    results = garden_resolved_crux(
        receipt,
        config=_CFG,
        claim_results={"claim-a": _pass_result("claim-a")},
    )
    assert results[0].status == STATUS_INSUFFICIENT_EVIDENCE


def test_summary_counts_mixed_findings() -> None:
    """run_gardening_pass summary counts all status buckets correctly."""
    # 2 resolved receipts: 1 stale, 1 contradiction
    stale_receipt = _receipt(crux_id="r-stale", affected_claims=["claim-stale"])
    contradiction_receipt = _receipt(crux_id="r-contra", affected_claims=["claim-ok"])
    coherence_issue = CoherenceIssue(
        kind=IncoherenceKind.CONTRADICTION,
        belief_ids=("claim-ok",),
        detail="conflict",
        severity="error",
    )
    # 2 outstanding entries: 1 fragility_shift, 1 healthy
    shift_entry = _entry(crux_id="o-shift")
    healthy_entry = _entry(crux_id="o-healthy")

    report = run_gardening_pass(
        [stale_receipt, contradiction_receipt],
        [shift_entry, healthy_entry],
        claim_results={
            "claim-stale": _stale_result("claim-stale"),
            "claim-ok": _pass_result("claim-ok"),
        },
        coherence_issues=[coherence_issue],
        fragility_scores={
            "o-shift": (0.3, 0.6),  # delta 0.3 >= threshold
            "o-healthy": (0.5, 0.52),  # delta 0.02 < threshold
        },
    )
    assert report.summary["stale_evidence"] == 1
    assert report.summary["new_contradiction"] == 1
    assert report.summary["fragility_shift"] == 1
    assert report.summary["healthy"] == 1
    assert report.summary["needs_followup"] == 0
