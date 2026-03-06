"""Tests for CLB-011 (taint propagation) and CLB-012 (external verifier gate).

Covers:
- propagate_taint merges and deduplicates
- TaintChecker.has_taint with clean and tainted bundles
- TaintChecker.collect_taint_summary across stages
- ExternalVerifierGate.requires_external_review thresholds
- ExternalVerifierGate.record_verification
"""

from __future__ import annotations

import pytest

from aragora.pipeline.backbone_contracts import (
    DeliberationBundle,
    ExecutionAttemptRecord,
    IntakeBundle,
    ReceiptEnvelope,
    SpecBundle,
    TaintChecker,
    propagate_taint,
)
from aragora.pipeline.external_verifier import ExternalVerifierGate


# ── propagate_taint ──────────────────────────────────────────────────────


class TestPropagateTaint:
    def test_merges_two_lists(self) -> None:
        result = propagate_taint(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]

    def test_deduplicates_preserving_order(self) -> None:
        result = propagate_taint(["a", "b"], ["b", "c", "a"])
        assert result == ["a", "b", "c"]

    def test_empty_inputs(self) -> None:
        assert propagate_taint([], []) == []
        assert propagate_taint([], None) == []

    def test_none_new_taint(self) -> None:
        result = propagate_taint(["x"], None)
        assert result == ["x"]

    def test_strips_whitespace(self) -> None:
        result = propagate_taint(["  a  "], ["  b  ", "a"])
        assert result == ["a", "b"]

    def test_skips_blank_strings(self) -> None:
        result = propagate_taint(["a", "", "  "], ["b"])
        assert result == ["a", "b"]

    def test_source_only(self) -> None:
        result = propagate_taint(["upstream_flag"])
        assert result == ["upstream_flag"]


# ── TaintChecker.has_taint ───────────────────────────────────────────────


class TestTaintCheckerHasTaint:
    def test_clean_intake_bundle(self) -> None:
        bundle = IntakeBundle(source_kind="test", raw_intent="do stuff")
        assert TaintChecker.has_taint(bundle) is False

    def test_tainted_intake_bundle(self) -> None:
        bundle = IntakeBundle(
            source_kind="test",
            raw_intent="do stuff",
            taint_flags=["external_unverified"],
        )
        assert TaintChecker.has_taint(bundle) is True

    def test_clean_spec_bundle(self) -> None:
        bundle = SpecBundle(title="T", problem_statement="P")
        assert TaintChecker.has_taint(bundle) is False

    def test_tainted_spec_bundle(self) -> None:
        bundle = SpecBundle(
            title="T",
            problem_statement="P",
            taint_flags=["low_confidence_source"],
        )
        assert TaintChecker.has_taint(bundle) is True

    def test_clean_deliberation_bundle(self) -> None:
        bundle = DeliberationBundle(debate_id="d1", verdict="ok")
        assert TaintChecker.has_taint(bundle) is False

    def test_tainted_deliberation_bundle(self) -> None:
        bundle = DeliberationBundle(
            debate_id="d1",
            verdict="ok",
            taint_flags=["no_consensus"],
        )
        assert TaintChecker.has_taint(bundle) is True

    def test_clean_execution_attempt(self) -> None:
        bundle = ExecutionAttemptRecord(attempt_id="a1")
        assert TaintChecker.has_taint(bundle) is False

    def test_tainted_execution_attempt(self) -> None:
        bundle = ExecutionAttemptRecord(
            attempt_id="a1",
            taint_flags=["external_dependency"],
        )
        assert TaintChecker.has_taint(bundle) is True

    def test_object_without_taint_flags(self) -> None:
        """Arbitrary objects that lack taint_flags are treated as clean."""

        class Plain:
            pass

        assert TaintChecker.has_taint(Plain()) is False


# ── TaintChecker.collect_taint_summary ───────────────────────────────────


class TestTaintCheckerCollectSummary:
    def test_all_clean_stages(self) -> None:
        intake = IntakeBundle(source_kind="test", raw_intent="x")
        spec = SpecBundle(title="T", problem_statement="P")
        delib = DeliberationBundle(debate_id="d", verdict="v")
        execution = ExecutionAttemptRecord(attempt_id="a")

        summary = TaintChecker.collect_taint_summary(
            intake=intake,
            spec=spec,
            deliberation=delib,
            execution=execution,
        )

        assert summary["tainted"] is False
        assert summary["flags"] == []
        assert summary["per_stage"] == {}

    def test_mixed_tainted_stages(self) -> None:
        intake = IntakeBundle(
            source_kind="test",
            raw_intent="x",
            taint_flags=["external_note"],
        )
        spec = SpecBundle(title="T", problem_statement="P")
        execution = ExecutionAttemptRecord(
            attempt_id="a",
            taint_flags=["external_dependency"],
        )

        summary = TaintChecker.collect_taint_summary(
            intake=intake,
            spec=spec,
            execution=execution,
        )

        assert summary["tainted"] is True
        assert summary["flags"] == ["external_note", "external_dependency"]
        assert "intake" in summary["per_stage"]
        assert "execution" in summary["per_stage"]
        assert "spec" not in summary["per_stage"]

    def test_verification_stage_reads_taint_summary(self) -> None:
        envelope = ReceiptEnvelope(
            receipt_id="r1",
            artifact_hash="h",
            verdict="pass",
            taint_summary={"tainted": True, "flags": ["post_verify_flag"]},
        )

        summary = TaintChecker.collect_taint_summary(verification=envelope)

        assert summary["tainted"] is True
        assert "post_verify_flag" in summary["flags"]
        assert "verification" in summary["per_stage"]

    def test_deduplicates_across_stages(self) -> None:
        intake = IntakeBundle(
            source_kind="test",
            raw_intent="x",
            taint_flags=["shared_flag"],
        )
        spec = SpecBundle(
            title="T",
            problem_statement="P",
            taint_flags=["shared_flag", "spec_only"],
        )

        summary = TaintChecker.collect_taint_summary(intake=intake, spec=spec)

        assert summary["flags"] == ["shared_flag", "spec_only"]

    def test_all_none_stages(self) -> None:
        summary = TaintChecker.collect_taint_summary()
        assert summary["tainted"] is False
        assert summary["flags"] == []
        assert summary["per_stage"] == {}


# ── DeliberationBundle taint_flags field ─────────────────────────────────


class TestDeliberationBundleTaint:
    def test_taint_flags_default_empty(self) -> None:
        bundle = DeliberationBundle(debate_id="d", verdict="v")
        assert bundle.taint_flags == []

    def test_from_debate_result_passes_taint(self) -> None:
        from types import SimpleNamespace

        result = SimpleNamespace(
            debate_id="d1",
            task="t",
            final_answer="a",
            confidence=0.6,
            consensus_reached=True,
            consensus_strength="strong",
            consensus_variance=0.5,
            dissenting_views=[],
            participants=["x"],
            per_agent_similarity={"x": 0.9},
            convergence_status="converged",
            debate_cruxes=[],
            metadata={},
        )

        bundle = DeliberationBundle.from_debate_result(result, taint_flags=["debate_shortcut"])
        assert bundle.taint_flags == ["debate_shortcut"]

    def test_taint_flags_in_to_dict(self) -> None:
        bundle = DeliberationBundle(
            debate_id="d",
            verdict="v",
            taint_flags=["flag1"],
        )
        d = bundle.to_dict()
        assert d["taint_flags"] == ["flag1"]


# ── ExternalVerifierGate.requires_external_review ────────────────────────


class TestExternalVerifierRequiresReview:
    def test_low_impact_does_not_require_review(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("low") is False

    def test_medium_impact_does_not_require_review_default(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("medium") is False

    def test_high_impact_requires_review_default(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("high") is True

    def test_critical_impact_requires_review(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("critical") is True

    def test_custom_threshold_medium(self) -> None:
        gate = ExternalVerifierGate(impact_threshold="medium")
        assert gate.requires_external_review("low") is False
        assert gate.requires_external_review("medium") is True
        assert gate.requires_external_review("high") is True

    def test_custom_threshold_low(self) -> None:
        gate = ExternalVerifierGate(impact_threshold="low")
        assert gate.requires_external_review("low") is True

    def test_taint_trigger_forces_review(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("low", taint_flags=["external_unverified"]) is True

    def test_custom_taint_trigger(self) -> None:
        gate = ExternalVerifierGate(taint_triggers=["risky_source"])
        assert gate.requires_external_review("low", taint_flags=["risky_source"]) is True
        assert gate.requires_external_review("low", taint_flags=["safe_flag"]) is False

    def test_unknown_impact_level_only_checks_taint(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("unknown") is False
        assert gate.requires_external_review("unknown", taint_flags=["external_unverified"]) is True

    def test_invalid_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="impact_threshold must be one of"):
            ExternalVerifierGate(impact_threshold="extreme")

    def test_case_insensitive_impact(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("HIGH") is True
        assert gate.requires_external_review("  High  ") is True

    def test_no_taint_flags_none(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("low", taint_flags=None) is False

    def test_empty_taint_flags(self) -> None:
        gate = ExternalVerifierGate()
        assert gate.requires_external_review("low", taint_flags=[]) is False


# ── ExternalVerifierGate.record_verification ─────────────────────────────


class TestExternalVerifierRecord:
    def test_record_approved(self) -> None:
        gate = ExternalVerifierGate()
        result = gate.record_verification(
            verifier_id="human-alice",
            approved=True,
            notes="LGTM",
        )

        assert result["verifier_id"] == "human-alice"
        assert result["approved"] is True
        assert result["notes"] == "LGTM"
        assert "recorded_at" in result
        assert len(gate.verifications) == 1

    def test_record_rejected(self) -> None:
        gate = ExternalVerifierGate()
        result = gate.record_verification(
            verifier_id="bot-reviewer",
            approved=False,
            notes="Risk too high",
        )

        assert result["approved"] is False
        assert result["notes"] == "Risk too high"

    def test_multiple_verifications_accumulate(self) -> None:
        gate = ExternalVerifierGate()
        gate.record_verification("v1", approved=False, notes="no")
        gate.record_verification("v2", approved=True, notes="yes")

        assert len(gate.verifications) == 2
        assert gate.verifications[0].verifier_id == "v1"
        assert gate.verifications[1].verifier_id == "v2"

    def test_default_empty_notes(self) -> None:
        gate = ExternalVerifierGate()
        result = gate.record_verification("v1", approved=True)
        assert result["notes"] == ""

    def test_strips_verifier_id_whitespace(self) -> None:
        gate = ExternalVerifierGate()
        result = gate.record_verification("  spaced  ", approved=True)
        assert result["verifier_id"] == "spaced"

    def test_recorded_at_is_utc_iso(self) -> None:
        gate = ExternalVerifierGate()
        result = gate.record_verification("v1", approved=True)
        # Should be parseable and contain UTC offset
        assert "T" in result["recorded_at"]
        assert "+" in result["recorded_at"] or "Z" in result["recorded_at"]
