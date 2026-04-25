"""Unit tests for DIC-23 Dialectical Runtime Loop (aragora.epistemic.runtime_loop)."""

from __future__ import annotations

import pytest

from aragora.epistemic.decay_monitor import DecayReason, DecaySignal
from aragora.epistemic.quarantine_policy import QuarantinePolicy
from aragora.epistemic.runtime_loop import (
    DialecticalEvent,
    DialecticalRuntimeError,
    dialectical_runtime_enabled,
    run_dialectical_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    *,
    unit_id: str = "unit.test",
    integrity_score: float = 0.8,
    recommended_action: str = "report_only",
    reasons: list[DecayReason] | None = None,
) -> DecaySignal:
    return DecaySignal(
        code_unit_id=unit_id,
        integrity_score=integrity_score,
        reasons=reasons or [],
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ARAGORA_DIALECTICAL_RUNTIME_ENABLED", raising=False)
        assert not dialectical_runtime_enabled()

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_truthy_values_enable_runtime(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("ARAGORA_DIALECTICAL_RUNTIME_ENABLED", value)
        assert dialectical_runtime_enabled() is True

    def test_raises_when_flag_off_and_require_enabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ARAGORA_DIALECTICAL_RUNTIME_ENABLED", raising=False)
        with pytest.raises(DialecticalRuntimeError, match="ARAGORA_DIALECTICAL_RUNTIME_ENABLED"):
            run_dialectical_loop(_signal())

    def test_require_enabled_false_bypasses_flag(self):
        event = run_dialectical_loop(_signal(), require_enabled=False)
        assert isinstance(event, DialecticalEvent)

    def test_flag_on_allows_call(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARAGORA_DIALECTICAL_RUNTIME_ENABLED", "1")
        event = run_dialectical_loop(_signal())
        assert isinstance(event, DialecticalEvent)


# ---------------------------------------------------------------------------
# Report-only (default) trace
# ---------------------------------------------------------------------------


class TestReportOnlyTrace:
    def test_event_fields_populated(self):
        sig = _signal(unit_id="unit.alpha", integrity_score=0.75)
        event = run_dialectical_loop(sig, require_enabled=False)

        assert event.code_unit_id == "unit.alpha"
        assert event.integrity_score == 0.75
        assert event.recommended_action == "report_only"
        assert event.repair_spec is None
        assert event.crux_probe_skipped is True
        assert event.prior_receipt_ids == ()
        assert event.created_at  # non-empty timestamp

    def test_event_id_has_drt_prefix(self):
        event = run_dialectical_loop(_signal(), require_enabled=False)
        assert event.event_id.startswith("drt_")

    def test_quarantine_action_for_healthy_unit(self):
        # integrity_score=0.8 → above default fail_closed_threshold (0.4)
        # recommended_action="report_only" → policy resolves to report_only
        event = run_dialectical_loop(_signal(integrity_score=0.8), require_enabled=False)
        assert event.quarantine_action == "report_only"

    def test_prior_receipt_ids_preserved(self):
        receipts = ["rcpt-001", "rcpt-002"]
        event = run_dialectical_loop(_signal(), prior_receipt_ids=receipts, require_enabled=False)
        assert event.prior_receipt_ids == tuple(receipts)

    def test_metadata_preserved(self):
        event = run_dialectical_loop(_signal(), metadata={"source": "test"}, require_enabled=False)
        assert event.metadata == {"source": "test"}

    def test_metadata_outer_mapping_is_immutable(self):
        event = run_dialectical_loop(_signal(), metadata={"source": "test"}, require_enabled=False)

        with pytest.raises(TypeError):
            event.metadata["source"] = "mutated"  # type: ignore[index]

    def test_to_dict_is_json_friendly(self):
        import json

        event = run_dialectical_loop(_signal(), require_enabled=False)
        d = event.to_dict()
        serialized = json.dumps(d)
        assert "drt_" in serialized
        assert "report_only" in serialized

    def test_to_dict_repair_spec_is_none(self):
        event = run_dialectical_loop(_signal(), require_enabled=False)
        assert event.to_dict()["repair_spec"] is None


# ---------------------------------------------------------------------------
# Repair-proposal path
# ---------------------------------------------------------------------------


class TestRepairProposal:
    def test_repair_spec_none_when_flag_false(self):
        sig = _signal(integrity_score=0.1, recommended_action="repair_required")
        event = run_dialectical_loop(sig, enable_repair_proposal=False, require_enabled=False)
        assert event.repair_spec is None

    def test_repair_spec_absent_when_policy_escalates_to_fail_closed(self):
        # score=0.1 is below default fail_closed_threshold=0.4, so the policy
        # escalates to fail_closed — NOT repair_required.  repair_spec must stay
        # None even when enable_repair_proposal=True.
        sig = _signal(integrity_score=0.1, recommended_action="repair_required")
        event = run_dialectical_loop(
            sig,
            code_unit_class="default",
            enable_repair_proposal=True,
            require_enabled=False,
        )
        assert event.quarantine_action == "fail_closed"
        assert event.repair_spec is None

    def test_repair_spec_produced_with_custom_policy(self):
        from aragora.epistemic.quarantine_policy import EscalationMap

        sig = _signal(integrity_score=0.5, recommended_action="repair_required")
        # Custom policy: repair_required stays repair_required, threshold=0.3
        policy = QuarantinePolicy(
            code_unit_class="custom",
            escalation_map=EscalationMap(
                report_only="report_only",
                repair_required="repair_required",
                fail_closed="fail_closed",
            ),
            fail_closed_threshold=0.3,
        )
        event = run_dialectical_loop(
            sig, policy=policy, enable_repair_proposal=True, require_enabled=False
        )
        assert event.repair_spec is not None
        assert event.repair_spec.code_unit_id == "unit.test"
        assert event.repair_spec.repair_kind == "report_only"
        assert event.quarantine_action == "repair_required"


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


class TestPolicyResolution:
    def test_explicit_policy_overrides_class(self):
        from aragora.epistemic.quarantine_policy import EscalationMap

        sig = _signal(integrity_score=0.9, recommended_action="report_only")
        policy = QuarantinePolicy(
            code_unit_class="strict",
            escalation_map=EscalationMap(),
            fail_closed_threshold=0.95,  # very strict: 0.9 → fail_closed
        )
        event = run_dialectical_loop(sig, policy=policy, require_enabled=False)
        assert event.quarantine_action == "fail_closed"

    def test_code_unit_class_live_dispatch(self):
        # live_dispatch policy has fail_closed_threshold=0.6; score=0.5 → fail_closed
        sig = _signal(integrity_score=0.5, recommended_action="report_only")
        event = run_dialectical_loop(sig, code_unit_class="live_dispatch", require_enabled=False)
        assert event.quarantine_action == "fail_closed"

    def test_code_unit_class_report_surface(self):
        # report_surface policy has fail_closed_threshold=0.3; score=0.5 → degrade
        sig = _signal(integrity_score=0.5, recommended_action="repair_required")
        event = run_dialectical_loop(sig, code_unit_class="report_surface", require_enabled=False)
        assert event.quarantine_action in {"degrade", "repair_required", "report_only"}


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_keys(self):
        event = run_dialectical_loop(_signal(), require_enabled=False)
        d = event.to_dict()
        expected_keys = {
            "event_id",
            "code_unit_id",
            "integrity_score",
            "recommended_action",
            "quarantine_action",
            "crux_probe_skipped",
            "repair_spec",
            "prior_receipt_ids",
            "created_at",
            "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_integrity_score_rounded(self):
        sig = _signal(integrity_score=0.123456789)
        event = run_dialectical_loop(sig, require_enabled=False)
        d = event.to_dict()
        assert d["integrity_score"] == round(0.123456789, 4)

    def test_prior_receipt_ids_is_immutable_tuple(self):
        receipts = ["r1", "r2"]
        event = run_dialectical_loop(_signal(), prior_receipt_ids=receipts, require_enabled=False)
        # Stored as an immutable tuple
        assert isinstance(event.prior_receipt_ids, tuple)
        assert event.prior_receipt_ids == ("r1", "r2")
        # to_dict() returns a list for JSON-serializability
        d = event.to_dict()
        assert d["prior_receipt_ids"] == ["r1", "r2"]
        # Mutation of the original list does not affect the event
        receipts.append("r3")
        assert event.prior_receipt_ids == ("r1", "r2")

    def test_with_repair_spec_in_dict(self):
        from aragora.epistemic.quarantine_policy import EscalationMap

        sig = _signal(integrity_score=0.5, recommended_action="repair_required")
        policy = QuarantinePolicy(
            code_unit_class="custom",
            escalation_map=EscalationMap(
                report_only="report_only",
                repair_required="repair_required",
                fail_closed="fail_closed",
            ),
            fail_closed_threshold=0.3,
        )
        event = run_dialectical_loop(
            sig, policy=policy, enable_repair_proposal=True, require_enabled=False
        )
        d = event.to_dict()
        assert d["repair_spec"] is not None
        assert isinstance(d["repair_spec"], dict)
        assert "spec_id" in d["repair_spec"]
