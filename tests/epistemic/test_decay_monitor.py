"""Unit tests for DIC-20 decay monitor (aragora.epistemic.decay_monitor)."""

from __future__ import annotations

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus
from aragora.epistemic.decay_monitor import (
    evaluate_unit,
)
from aragora.epistemic.proof_unit import (
    DecayPolicy,
    FallbackPolicy,
    ProofCarryingCodeUnit,
)


def _make_unit(
    *,
    claims: list[str] | None = None,
    linked_crux_ids: list[str] | None = None,
    decision_receipts: list[str] | None = None,
    failed_claim_policy: str = "report_only",
    stale_evidence_policy: str = "report_only",
    unresolved_crux_policy: str = "report_only",
) -> ProofCarryingCodeUnit:
    return ProofCarryingCodeUnit(
        code_unit_id="unit.test",
        symbol="tests.fake.function",
        source_path="tests/fake.py",
        owner="test",
        decision_receipts=decision_receipts if decision_receipts is not None else ["receipt-001"],
        claims=claims or [],
        assumptions=["Tests run in a clean environment."],
        verifiers=[],
        freshness_sla_hours=24,
        decay_policy=DecayPolicy(
            failed_claim=failed_claim_policy,
            stale_evidence=stale_evidence_policy,
            unresolved_crux=unresolved_crux_policy,
        ),
        fallback_policy=FallbackPolicy(default="report_only"),
        linked_crux_ids=linked_crux_ids or [],
    )


def _claim(claim_id: str, status: ClaimStatus, message: str = "") -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        status=status,
        message=message or status.value,
    )


class TestHealthyUnit:
    def test_no_claims_with_receipt_is_one(self):
        signal = evaluate_unit(_make_unit())
        assert signal.integrity_score == 1.0
        assert signal.reasons == []
        assert signal.recommended_action == "report_only"

    def test_passing_claims_no_deduction(self):
        unit = _make_unit(claims=["c1", "c2"])
        results = {
            "c1": _claim("c1", ClaimStatus.PASS),
            "c2": _claim("c2", ClaimStatus.PASS),
        }
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.integrity_score == 1.0
        assert not signal.reasons

    def test_unsupported_claim_no_deduction(self):
        unit = _make_unit(claims=["c1"])
        results = {"c1": _claim("c1", ClaimStatus.UNSUPPORTED)}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.integrity_score == 1.0

    def test_missing_claim_result_no_deduction(self):
        unit = _make_unit(claims=["c1"])
        signal = evaluate_unit(unit, claim_results={})
        assert signal.integrity_score == 1.0


class TestFailedClaim:
    def test_failed_claim_deducts_score(self):
        unit = _make_unit(claims=["c1"])
        results = {"c1": _claim("c1", ClaimStatus.FAIL, "assertion failed")}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.integrity_score < 1.0
        assert any(r.kind == "failed_claim" for r in signal.reasons)

    def test_failed_claim_carries_claim_id(self):
        unit = _make_unit(claims=["claim.foo"])
        results = {"claim.foo": _claim("claim.foo", ClaimStatus.FAIL)}
        signal = evaluate_unit(unit, claim_results=results)
        reason = next(r for r in signal.reasons if r.kind == "failed_claim")
        assert reason.claim_id == "claim.foo"

    def test_failed_claim_repair_required_policy(self):
        unit = _make_unit(claims=["c1"], failed_claim_policy="repair_required")
        results = {"c1": _claim("c1", ClaimStatus.FAIL)}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.recommended_action == "repair_required"

    def test_failed_claim_fail_closed_policy(self):
        unit = _make_unit(claims=["c1"], failed_claim_policy="fail_closed")
        results = {"c1": _claim("c1", ClaimStatus.FAIL)}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.recommended_action == "fail_closed"


class TestStaleEvidence:
    def test_stale_claim_creates_reason(self):
        unit = _make_unit(claims=["c1"])
        results = {"c1": _claim("c1", ClaimStatus.STALE, "evidence older than SLA")}
        signal = evaluate_unit(unit, claim_results=results)
        assert any(r.kind == "stale_evidence" for r in signal.reasons)
        assert signal.integrity_score < 1.0

    def test_stale_policy_propagates(self):
        unit = _make_unit(claims=["c1"], stale_evidence_policy="repair_required")
        results = {"c1": _claim("c1", ClaimStatus.STALE)}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.recommended_action == "repair_required"

    def test_stale_is_lighter_than_fail(self):
        unit_fail = _make_unit(claims=["c1"])
        unit_stale = _make_unit(claims=["c1"])
        s_fail = evaluate_unit(unit_fail, {"c1": _claim("c1", ClaimStatus.FAIL)})
        s_stale = evaluate_unit(unit_stale, {"c1": _claim("c1", ClaimStatus.STALE)})
        assert s_stale.integrity_score > s_fail.integrity_score


class TestUnresolvedCrux:
    def test_unresolved_crux_creates_reason(self):
        unit = _make_unit(linked_crux_ids=["crux.abc"])
        signal = evaluate_unit(unit, unresolved_crux_ids=frozenset({"crux.abc"}))
        assert any(r.kind == "unresolved_crux" for r in signal.reasons)
        assert signal.integrity_score < 1.0

    def test_resolved_crux_no_reason(self):
        unit = _make_unit(linked_crux_ids=["crux.abc"])
        signal = evaluate_unit(unit, unresolved_crux_ids=frozenset())
        assert not any(r.kind == "unresolved_crux" for r in signal.reasons)

    def test_crux_carries_crux_id(self):
        unit = _make_unit(linked_crux_ids=["crux.xyz"])
        signal = evaluate_unit(unit, unresolved_crux_ids=frozenset({"crux.xyz"}))
        reason = next(r for r in signal.reasons if r.kind == "unresolved_crux")
        assert reason.crux_id == "crux.xyz"

    def test_unresolved_crux_policy_fail_closed(self):
        unit = _make_unit(
            linked_crux_ids=["crux.y"],
            unresolved_crux_policy="fail_closed",
        )
        signal = evaluate_unit(unit, unresolved_crux_ids=frozenset({"crux.y"}))
        assert signal.recommended_action == "fail_closed"


class TestMissingReceipt:
    def test_empty_receipts_creates_reason(self):
        unit = _make_unit(decision_receipts=[])
        signal = evaluate_unit(unit)
        assert any(r.kind == "missing_receipt" for r in signal.reasons)
        assert signal.integrity_score < 1.0

    def test_receipt_present_no_reason(self):
        unit = _make_unit(decision_receipts=["r-001"])
        signal = evaluate_unit(unit)
        assert not any(r.kind == "missing_receipt" for r in signal.reasons)


class TestVerifierError:
    def test_error_status_creates_reason(self):
        unit = _make_unit(claims=["c1"])
        results = {"c1": _claim("c1", ClaimStatus.ERROR, "subprocess timeout")}
        signal = evaluate_unit(unit, claim_results=results)
        assert any(r.kind == "verifier_error" for r in signal.reasons)

    def test_error_uses_failed_claim_policy(self):
        unit = _make_unit(claims=["c1"], failed_claim_policy="repair_required")
        results = {"c1": _claim("c1", ClaimStatus.ERROR)}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.recommended_action == "repair_required"


class TestIntegrityScoreBounds:
    def test_score_never_below_zero(self):
        unit = _make_unit(
            claims=["c1", "c2", "c3", "c4"],
            linked_crux_ids=["crux.1", "crux.2"],
            decision_receipts=[],
        )
        results = {
            "c1": _claim("c1", ClaimStatus.FAIL),
            "c2": _claim("c2", ClaimStatus.FAIL),
            "c3": _claim("c3", ClaimStatus.STALE),
            "c4": _claim("c4", ClaimStatus.ERROR),
        }
        signal = evaluate_unit(
            unit,
            claim_results=results,
            unresolved_crux_ids=frozenset({"crux.1", "crux.2"}),
        )
        assert signal.integrity_score >= 0.0

    def test_fully_healthy_score_is_exactly_one(self):
        unit = _make_unit(claims=["c1"], decision_receipts=["r1"])
        results = {"c1": _claim("c1", ClaimStatus.PASS)}
        signal = evaluate_unit(unit, claim_results=results)
        assert signal.integrity_score == 1.0


class TestToDict:
    def test_required_keys_present(self):
        signal = evaluate_unit(_make_unit())
        d = signal.to_dict()
        assert set(d) >= {"code_unit_id", "integrity_score", "reasons", "recommended_action"}

    def test_reason_dict_has_kind_and_detail(self):
        unit = _make_unit(decision_receipts=[])
        signal = evaluate_unit(unit)
        reason_dict = signal.to_dict()["reasons"][0]
        assert "kind" in reason_dict
        assert "detail" in reason_dict

    def test_claim_id_in_reason_dict_when_set(self):
        unit = _make_unit(claims=["c1"])
        results = {"c1": _claim("c1", ClaimStatus.FAIL)}
        signal = evaluate_unit(unit, claim_results=results)
        reason_dicts = signal.to_dict()["reasons"]
        assert any("claim_id" in r for r in reason_dicts)

    def test_crux_id_not_in_reason_dict_when_unset(self):
        unit = _make_unit(decision_receipts=[])
        signal = evaluate_unit(unit)
        reason_dict = signal.to_dict()["reasons"][0]
        assert "crux_id" not in reason_dict

    def test_integrity_score_is_rounded(self):
        unit = _make_unit(claims=["c1"], decision_receipts=["r"])
        results = {"c1": _claim("c1", ClaimStatus.STALE)}
        signal = evaluate_unit(unit, claim_results=results)
        score = signal.to_dict()["integrity_score"]
        assert isinstance(score, float)
        assert str(score).count(".") == 1
        assert len(str(score).split(".")[1]) <= 4


# =============================================================================
# Round 2026-04-30c Phase E — DIC-19 multi-hop wiring into decay impact
# First live caller of ProofUnitConstraintGraph.multi_hop_impact_set (#6838).
# =============================================================================

from aragora.epistemic.constraint_graph import ProofUnitConstraintGraph
from aragora.epistemic.decay_monitor import compute_decay_impact_set


def _bare_unit(uid: str, claims: list[str] | None = None) -> ProofCarryingCodeUnit:
    """Minimal unit fixture for impact-set tests (avoids decay-policy noise)."""
    return ProofCarryingCodeUnit(
        code_unit_id=uid,
        symbol=f"tests.fake.{uid}",
        source_path=f"tests/{uid}.py",
        owner="test",
        decision_receipts=[f"r-{uid}"],
        claims=claims or [],
        assumptions=[],
        verifiers=[],
        freshness_sla_hours=24,
        decay_policy=DecayPolicy(),
        fallback_policy=FallbackPolicy(),
        linked_crux_ids=[],
    )


class TestComputeDecayImpactSet:
    def test_no_failing_claims_returns_empty(self):
        g = ProofUnitConstraintGraph([_bare_unit("a", claims=["x"])])
        assert compute_decay_impact_set(g, set()) == set()

    def test_single_hop_returns_direct_owners(self):
        g = ProofUnitConstraintGraph([_bare_unit("a", claims=["x"]), _bare_unit("b", claims=["y"])])
        assert compute_decay_impact_set(g, {"x"}) == {"a"}
        assert compute_decay_impact_set(g, {"x", "y"}) == {"a", "b"}

    def test_transitive_with_no_edges_equals_single_hop(self):
        """Backward-compat: graph constructed without edges -> same impact."""
        g = ProofUnitConstraintGraph([_bare_unit("a", claims=["x"]), _bare_unit("b", claims=["y"])])
        assert compute_decay_impact_set(g, {"x"}, transitive=True) == compute_decay_impact_set(
            g, {"x"}
        )

    def test_transitive_walks_dependency_edges(self):
        # B depends on A. If A's claim x fails, B is also impacted.
        g = ProofUnitConstraintGraph(
            [_bare_unit("a", claims=["x"]), _bare_unit("b")],
            dependency_edges=[("b", "a")],
        )
        assert compute_decay_impact_set(g, {"x"}) == {"a"}
        assert compute_decay_impact_set(g, {"x"}, transitive=True) == {"a", "b"}

    def test_transitive_chain_propagates(self):
        # C -> B -> A; failing claim x cascades to all three when transitive.
        g = ProofUnitConstraintGraph(
            [_bare_unit("a", claims=["x"]), _bare_unit("b"), _bare_unit("c")],
            dependency_edges=[("b", "a"), ("c", "b")],
        )
        assert compute_decay_impact_set(g, {"x"}, transitive=True) == {"a", "b", "c"}

    def test_max_depth_zero_equals_seed(self):
        g = ProofUnitConstraintGraph(
            [_bare_unit("a", claims=["x"]), _bare_unit("b"), _bare_unit("c")],
            dependency_edges=[("b", "a"), ("c", "b")],
        )
        assert compute_decay_impact_set(g, {"x"}, transitive=True, max_depth=0) == {"a"}

    def test_max_depth_one_limits_propagation(self):
        g = ProofUnitConstraintGraph(
            [_bare_unit("a", claims=["x"]), _bare_unit("b"), _bare_unit("c")],
            dependency_edges=[("b", "a"), ("c", "b")],
        )
        assert compute_decay_impact_set(g, {"x"}, transitive=True, max_depth=1) == {"a", "b"}

    def test_max_depth_irrelevant_when_not_transitive(self):
        g = ProofUnitConstraintGraph(
            [_bare_unit("a", claims=["x"]), _bare_unit("b")],
            dependency_edges=[("b", "a")],
        )
        assert compute_decay_impact_set(g, {"x"}, transitive=False, max_depth=999) == {"a"}

    def test_unknown_claim_returns_empty(self):
        g = ProofUnitConstraintGraph([_bare_unit("a", claims=["x"])])
        assert compute_decay_impact_set(g, {"unknown"}) == set()
        assert compute_decay_impact_set(g, {"unknown"}, transitive=True) == set()

    def test_exported_from_aragora_epistemic(self):
        """Public surface is reachable via the package."""
        import aragora.epistemic as ep

        assert hasattr(ep, "compute_decay_impact_set")
        assert "compute_decay_impact_set" in ep.__all__
