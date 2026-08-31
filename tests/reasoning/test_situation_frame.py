"""Tests for evidence/possibility/control envelopes and protected truncation."""

import json

from aragora.export.decision_receipt import ReceiptDissent
from aragora.reasoning.epistemics import EpistemicTag, KnowledgeState, ProvenanceClass
from aragora.reasoning.situation_frame import (
    ControlEnvelope,
    EvidenceEnvelope,
    EvidenceFact,
    PossibilityEnvelope,
    PossibilityResidual,
    SituationFrame,
    from_receipt_dissents,
    truncate_frame,
)
from aragora.work.affordances import ActionAffordance, AffordanceDisposition, CostVector


def _fact(fid: str) -> EvidenceFact:
    return EvidenceFact(
        fact_id=fid,
        statement=f"fact {fid}",
        tag=EpistemicTag(state=KnowledgeState.KNOWN, provenance=ProvenanceClass.OBSERVED),
        evidence_refs=[f"ref:{fid}"],
    )


def _residual(rid: str, severity: float) -> PossibilityResidual:
    return PossibilityResidual(
        residual_id=rid,
        description=f"alternative world {rid} " + "x" * 40,
        loss_severity=severity,
        source="agent-red",
    )


def _aff(aid: str, disposition: AffordanceDisposition) -> ActionAffordance:
    return ActionAffordance(
        affordance_id=aid,
        target="t",
        operation="op",
        reason_available="r",
        disposition=disposition,
        expected_gain="g",
        expected_value=1.0,
        cost=CostVector(),
        risk_tier=0,
        reversibility="reversible",
        required_capabilities=[],
        required_approvals=[],
        preconditions=[],
        invalidators=[],
        alternatives=[],
        expected_terminal_proof="p",
    )


def _frame(residuals, affordances=()) -> SituationFrame:
    return SituationFrame(
        anchor={"repo": "aragora", "commit": "1aa62825", "branch": "main", "clean": "true"},
        evidence=EvidenceEnvelope(facts=[_fact("f1")], certified_absences=[], assumptions=[]),
        possibility=PossibilityEnvelope(residuals=list(residuals)),
        control=ControlEnvelope(affordances=list(affordances)),
        generated_at=1000.0,
    )


class TestFromReceiptDissents:
    def test_maps_fields_and_preserves_severity(self):
        dissent = ReceiptDissent(
            agent="claude",
            type="safety",
            severity=0.9,
            reasons=["rollback path unproven"],
            alternative="stage behind a flag",
        )
        (residual,) = from_receipt_dissents([dissent])
        assert residual.loss_severity == 0.9
        assert residual.source == "claude"
        assert "rollback path unproven" in residual.description
        assert "stage behind a flag" in residual.description


class TestProtectedTruncation:
    def test_within_budget_drops_nothing_and_reports_truthful_bytes(self):
        frame = _frame([_residual("r1", 0.9)])
        out, report = truncate_frame(frame, budget_bytes=1_000_000)
        assert report.dropped_residuals == 0
        assert not report.over_budget
        assert report.emitted_bytes == len(
            json.dumps(out.to_dict(), separators=(",", ":")).encode("utf-8")
        )

    def test_low_severity_residuals_drop_first_high_severity_survive(self):
        low = [_residual(f"low{i}", 0.1) for i in range(30)]
        high = _residual("high", 0.95)
        frame = _frame([*low, high])
        tight = len(json.dumps(_frame([high]).to_dict(), separators=(",", ":")).encode()) + 200
        out, report = truncate_frame(frame, budget_bytes=tight)
        kept = {r.residual_id for r in out.possibility.residuals}
        assert "high" in kept  # the invariant: high-loss residuals cannot disappear
        assert report.dropped_residuals > 0
        assert report.protected_retained == 1

    def test_protected_overflow_is_reported_never_silently_dropped(self):
        """If protected content alone exceeds the budget, keep it and say so."""
        protected = [_residual(f"p{i}", 0.9) for i in range(50)]
        frame = _frame(protected)
        out, report = truncate_frame(frame, budget_bytes=300)
        assert len(out.possibility.residuals) == 50
        assert report.over_budget
        assert report.emitted_bytes > 300

    def test_blocked_affordances_survive_unavailable_drop(self):
        blocked = _aff("blocked", AffordanceDisposition.BLOCKED)
        unavailable = [_aff(f"u{i}", AffordanceDisposition.UNAVAILABLE) for i in range(30)]
        frame = _frame([], affordances=[blocked, *unavailable])
        tight = (
            len(json.dumps(_frame([], [blocked]).to_dict(), separators=(",", ":")).encode()) + 200
        )
        out, report = truncate_frame(frame, budget_bytes=tight)
        kept = {a.affordance_id for a in out.control.affordances}
        assert "blocked" in kept
        assert report.dropped_affordances > 0

    def test_evidence_facts_are_never_dropped(self):
        frame = _frame([_residual(f"r{i}", 0.1) for i in range(20)])
        out, _ = truncate_frame(frame, budget_bytes=100)
        assert [f.fact_id for f in out.evidence.facts] == ["f1"]
