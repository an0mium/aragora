"""Unit tests for DIC-22 repair-spec producer (aragora.epistemic.repair)."""

from __future__ import annotations

import pytest

from aragora.epistemic.decay_monitor import DecayReason, DecaySignal
from aragora.epistemic.repair import (
    RepairSpec,
    enable_repair_pipeline,
    propose_repair,
    repair_pipeline_enabled,
)


def _signal(*, reasons: list[DecayReason] | None = None) -> DecaySignal:
    return DecaySignal(
        code_unit_id="unit.test",
        integrity_score=0.4,
        reasons=reasons or [],
        recommended_action="repair_required",
    )


class TestFlagGate:
    def test_flag_off_by_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        assert not repair_pipeline_enabled()

    def test_flag_truthy_values(self, monkeypatch):
        for val in ("1", "true", "yes", "on"):
            monkeypatch.setenv("ARAGORA_REPAIR_PIPELINE_ENABLED", val)
            assert repair_pipeline_enabled()

    def test_enable_helper(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        enable_repair_pipeline()
        assert repair_pipeline_enabled()
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)


class TestNoHotswapInvariant:
    def test_live_swap_always_blocked(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_REPAIR_PIPELINE_ENABLED", "1")
        with pytest.raises(ValueError, match="permanently blocked"):
            propose_repair(_signal(), repair_kind="live_swap")  # type: ignore[arg-type]

    def test_live_swap_blocked_without_flag(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        with pytest.raises(ValueError, match="permanently blocked"):
            propose_repair(_signal(), repair_kind="live_swap")  # type: ignore[arg-type]

    def test_unknown_kind_raises(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_REPAIR_PIPELINE_ENABLED", "1")
        with pytest.raises(ValueError, match="not a known kind"):
            propose_repair(_signal(), repair_kind="rewrite_production")  # type: ignore[arg-type]

    def test_shadow_blocked_without_flag(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        with pytest.raises(ValueError, match="ARAGORA_REPAIR_PIPELINE_ENABLED"):
            propose_repair(_signal(), repair_kind="shadow_candidate")

    def test_pr_candidate_blocked_without_flag(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        with pytest.raises(ValueError, match="ARAGORA_REPAIR_PIPELINE_ENABLED"):
            propose_repair(_signal(), repair_kind="pr_candidate")


class TestReportOnly:
    def test_default_is_report_only(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        spec = propose_repair(_signal())
        assert spec.repair_kind == "report_only"

    def test_report_only_has_empty_provenance_hash(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        assert propose_repair(_signal()).provenance_hash == ""


class TestProvenanceHash:
    def test_shadow_carries_64char_hash(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_REPAIR_PIPELINE_ENABLED", "1")
        spec = propose_repair(_signal(), repair_kind="shadow_candidate")
        assert len(spec.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in spec.provenance_hash)

    def test_pr_candidate_carries_hash(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_REPAIR_PIPELINE_ENABLED", "1")
        spec = propose_repair(_signal(), repair_kind="pr_candidate")
        assert len(spec.provenance_hash) == 64


class TestLinkedFields:
    def test_claims_and_cruxes_extracted_from_reasons(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        reasons = [
            DecayReason(kind="failed_claim", detail="x", claim_id="claim.a"),
            DecayReason(kind="unresolved_crux", detail="y", crux_id="crux.z"),
        ]
        spec = propose_repair(_signal(reasons=reasons))
        assert "claim.a" in spec.linked_claims
        assert "crux.z" in spec.linked_crux_ids

    def test_explicit_overrides_default(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        reasons = [DecayReason(kind="failed_claim", detail="x", claim_id="auto")]
        spec = propose_repair(_signal(reasons=reasons), linked_claims=["manual"])
        assert spec.linked_claims == ["manual"]
        assert "auto" not in spec.linked_claims


class TestSerialization:
    def test_to_dict_round_trips_key_fields(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPAIR_PIPELINE_ENABLED", raising=False)
        spec = propose_repair(
            _signal(),
            validation_commands=["pytest tests/"],
            receipt_context={"receipt_id": "rec-1"},
        )
        d = spec.to_dict()
        assert d["repair_kind"] == "report_only"
        assert d["validation_commands"] == ["pytest tests/"]
        assert d["receipt_context"] == {"receipt_id": "rec-1"}
        assert "decay_signal" in d
        assert d["decay_signal"]["code_unit_id"] == "unit.test"
        assert d["spec_id"].startswith("repair-")
        assert len(d["spec_id"]) == len("repair-") + 16
