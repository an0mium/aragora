"""Tests for DIC-17 × DIC-20 bridge: propose_followup_for_decay_signal.

Flag-gated default OFF. No live queue effect.
Advances: DIC-17 (#6027) — decay signal → follow-up bridge.
"""

from __future__ import annotations

import pytest

from aragora.epistemic.decay_monitor import DecayReason, DecaySignal
from aragora.epistemic.followup import (
    DEFAULT_DECAY_INTEGRITY_THRESHOLD,
    FollowupProposal,
    propose_followup_for_decay_signal,
)


def _signal(
    code_unit_id: str = "test.unit",
    integrity_score: float = 0.5,
    recommended_action: str = "repair_required",
    reasons: list[DecayReason] | None = None,
) -> DecaySignal:
    return DecaySignal(
        code_unit_id=code_unit_id,
        integrity_score=integrity_score,
        recommended_action=recommended_action,
        reasons=reasons or [],
    )


class TestDecaySignalFilteringThreshold:
    def test_above_threshold_returns_none(self) -> None:
        sig = _signal(integrity_score=0.95, recommended_action="repair_required")
        assert propose_followup_for_decay_signal(sig) is None

    def test_exactly_at_threshold_returns_none(self) -> None:
        sig = _signal(
            integrity_score=DEFAULT_DECAY_INTEGRITY_THRESHOLD,
            recommended_action="repair_required",
        )
        assert propose_followup_for_decay_signal(sig) is None

    def test_below_threshold_report_only_returns_none(self) -> None:
        sig = _signal(integrity_score=0.3, recommended_action="report_only")
        assert propose_followup_for_decay_signal(sig) is None

    def test_below_threshold_repair_required_produces_proposal(self) -> None:
        sig = _signal(integrity_score=0.5, recommended_action="repair_required")
        proposal = propose_followup_for_decay_signal(sig)
        assert proposal is not None
        assert isinstance(proposal, FollowupProposal)

    def test_below_threshold_fail_closed_produces_proposal(self) -> None:
        sig = _signal(integrity_score=0.2, recommended_action="fail_closed")
        proposal = propose_followup_for_decay_signal(sig)
        assert proposal is not None

    def test_custom_threshold_higher_triggers_proposal(self) -> None:
        sig = _signal(integrity_score=0.85, recommended_action="repair_required")
        # default threshold (0.7) → None because 0.85 >= 0.7
        assert propose_followup_for_decay_signal(sig) is None
        # custom threshold 0.9 → proposal because 0.85 < 0.9
        proposal = propose_followup_for_decay_signal(sig, integrity_threshold=0.9)
        assert proposal is not None

    def test_invalid_threshold_raises(self) -> None:
        sig = _signal()
        with pytest.raises(ValueError, match="integrity_threshold"):
            propose_followup_for_decay_signal(sig, integrity_threshold=1.5)


class TestDecaySignalProposalShape:
    def _make(self, **kwargs: object) -> FollowupProposal:
        sig = _signal(**kwargs)  # type: ignore[arg-type]
        p = propose_followup_for_decay_signal(sig, integrity_threshold=0.9)
        assert p is not None
        return p

    def test_source_kind_is_decay_signal(self) -> None:
        p = self._make()
        assert p.source_kind == "decay_signal"

    def test_title_contains_unit_id(self) -> None:
        p = propose_followup_for_decay_signal(
            _signal(code_unit_id="my.proof.unit", integrity_score=0.5),
        )
        assert p is not None
        assert "my.proof.unit" in p.title

    def test_body_contains_integrity_score(self) -> None:
        sig = _signal(integrity_score=0.42, recommended_action="repair_required")
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "0.42" in p.body

    def test_fail_closed_urgency_in_body(self) -> None:
        sig = _signal(integrity_score=0.3, recommended_action="fail_closed")
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "fail-closed" in p.body

    def test_repair_required_urgency_in_body(self) -> None:
        sig = _signal(integrity_score=0.5, recommended_action="repair_required")
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "repair required" in p.body

    def test_reasons_listed_in_body(self) -> None:
        reasons = [
            DecayReason(kind="failed_claim", detail="Claim X failed", claim_id="claim_abc"),
            DecayReason(kind="unresolved_crux", detail="Crux Y open", crux_id="crux_xyz"),
        ]
        sig = _signal(integrity_score=0.4, reasons=reasons)
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "failed_claim" in p.body
        assert "claim_abc" in p.body
        assert "unresolved_crux" in p.body
        assert "crux_xyz" in p.body

    def test_unit_source_path_in_body_when_provided(self) -> None:
        sig = _signal(integrity_score=0.3)
        p = propose_followup_for_decay_signal(sig, unit_source_path="aragora/proof/unit.py")
        assert p is not None
        assert "aragora/proof/unit.py" in p.body

    def test_source_path_absent_from_body_when_omitted(self) -> None:
        sig = _signal(integrity_score=0.3)
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "source_path" not in p.body


class TestDecaySignalQueueGovernance:
    def test_boss_ready_never_in_labels(self) -> None:
        sig = _signal(integrity_score=0.1, recommended_action="fail_closed")
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "boss-ready" not in p.labels

    def test_epistemic_label_always_present(self) -> None:
        sig = _signal(integrity_score=0.5)
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "epistemic" in p.labels

    def test_decay_label_always_present(self) -> None:
        sig = _signal(integrity_score=0.5)
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "decay" in p.labels

    def test_extra_labels_merged_without_boss_ready(self) -> None:
        sig = _signal(integrity_score=0.5)
        p = propose_followup_for_decay_signal(
            sig, extra_labels=("priority-p1", "boss-ready")
        )
        assert p is not None
        assert "priority-p1" in p.labels
        assert "boss-ready" not in p.labels

    def test_queue_policy_in_body(self) -> None:
        sig = _signal(integrity_score=0.3)
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert "MUST NOT carry `boss-ready`" in p.body

    def test_source_key_is_deterministic(self) -> None:
        sig = _signal(code_unit_id="stable.unit", integrity_score=0.3)
        p1 = propose_followup_for_decay_signal(sig)
        p2 = propose_followup_for_decay_signal(sig)
        assert p1 is not None and p2 is not None
        assert p1.source_key == p2.source_key

    def test_different_units_give_different_source_keys(self) -> None:
        p1 = propose_followup_for_decay_signal(
            _signal(code_unit_id="unit.a", integrity_score=0.3)
        )
        p2 = propose_followup_for_decay_signal(
            _signal(code_unit_id="unit.b", integrity_score=0.3)
        )
        assert p1 is not None and p2 is not None
        assert p1.source_key != p2.source_key

    def test_provenance_has_expected_keys(self) -> None:
        sig = _signal(code_unit_id="prov.unit", integrity_score=0.4)
        p = propose_followup_for_decay_signal(sig)
        assert p is not None
        assert p.provenance["code_unit_id"] == "prov.unit"
        assert "integrity_score" in p.provenance
        assert "recommended_action" in p.provenance
        assert "reason_kinds" in p.provenance
