"""Tests for aragora.epistemic.proof_unit (DIC-19 / #6030).

Pure schema/validation — no network, no subprocess, no queue mutation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.epistemic.proof_unit import (
    DecayPolicy,
    FallbackPolicy,
    ProofCarryingCodeUnit,
    load_proof_unit,
    load_proof_unit_from_yaml,
)

PROOF_UNITS_DIR = Path(__file__).parents[2] / "docs" / "status" / "proof_units"


def _unit(**overrides: object) -> dict:
    base: dict = {
        "code_unit_id": "test.unit.alpha",
        "symbol": "aragora.epistemic.proof_unit.load_proof_unit",
        "source_path": "aragora/epistemic/proof_unit.py",
        "owner": "epistemic-ci",
        "decision_receipts": ["receipt.test.001"],
        "claims": ["b0.benchmark_truth.complete_current_corpus"],
        "assumptions": ["Assumption A holds."],
        "verifiers": [{"kind": "command", "command": "echo ok"}],
        "freshness_sla_hours": 24,
        "decay_policy": {
            "failed_claim": "report_only",
            "stale_evidence": "report_only",
            "unresolved_crux": "report_only",
        },
        "fallback_policy": {"default": "fail_closed", "operator_message": "Stop."},
    }
    base.update(overrides)
    return base


class TestLoad:
    def test_load_round_trips_code_unit_id(self) -> None:
        assert load_proof_unit(_unit()).code_unit_id == "test.unit.alpha"

    def test_linked_crux_ids_defaults_to_empty(self) -> None:
        assert load_proof_unit(_unit()).linked_crux_ids == []

    def test_freshness_sla_coerced_to_int(self) -> None:
        assert load_proof_unit(_unit(freshness_sla_hours="48")).freshness_sla_hours == 48

    def test_decay_and_fallback_typed(self) -> None:
        unit = load_proof_unit(_unit())
        assert isinstance(unit.decay_policy, DecayPolicy)
        assert isinstance(unit.fallback_policy, FallbackPolicy)


class TestValidation:
    def test_valid_unit_has_no_errors(self) -> None:
        assert load_proof_unit(_unit()).validate() == []

    def test_empty_code_unit_id_invalid(self) -> None:
        errors = load_proof_unit(_unit(code_unit_id="")).validate()
        assert any("code_unit_id" in e for e in errors)

    def test_zero_freshness_sla_invalid(self) -> None:
        errors = load_proof_unit(_unit(freshness_sla_hours=0)).validate()
        assert any("freshness_sla_hours" in e for e in errors)

    def test_invalid_decay_action_reported(self) -> None:
        data = _unit()
        data["decay_policy"]["failed_claim"] = "do_nothing_forever"
        errors = load_proof_unit(data).validate()
        assert any("decay_policy.failed_claim" in e for e in errors)

    def test_invalid_fallback_action_reported(self) -> None:
        data = _unit()
        data["fallback_policy"]["default"] = "yolo"
        errors = load_proof_unit(data).validate()
        assert any("fallback_policy.default" in e for e in errors)

    def test_all_valid_decay_actions_accepted(self) -> None:
        for action in ("report_only", "repair_required", "fail_closed"):
            data = _unit()
            data["decay_policy"]["failed_claim"] = action
            assert load_proof_unit(data).validate() == []

    def test_all_valid_fallback_actions_accepted(self) -> None:
        for action in ("fail_closed", "degrade", "report_only"):
            data = _unit()
            data["fallback_policy"]["default"] = action
            assert load_proof_unit(data).validate() == []


class TestSerialization:
    def test_to_dict_has_expected_keys(self) -> None:
        d = load_proof_unit(_unit()).to_dict()
        assert {
            "code_unit_id",
            "symbol",
            "source_path",
            "owner",
            "decision_receipts",
            "claims",
            "assumptions",
            "verifiers",
            "freshness_sla_hours",
            "decay_policy",
            "fallback_policy",
            "linked_crux_ids",
        } == set(d)

    def test_roundtrip_via_load(self) -> None:
        original = load_proof_unit(_unit())
        reloaded = load_proof_unit(original.to_dict())
        assert reloaded.code_unit_id == original.code_unit_id
        assert reloaded.decay_policy.failed_claim == original.decay_policy.failed_claim


class TestYamlLoading:
    def test_load_proof_first_shift_example(self) -> None:
        path = PROOF_UNITS_DIR / "proof_first_shift.yaml"
        assert path.exists(), f"example fixture missing: {path}"
        unit = load_proof_unit_from_yaml(path)
        assert unit.code_unit_id == "proof_first.shift.green_criteria"
        assert "b0.benchmark_truth.complete_current_corpus" in unit.claims
        assert unit.decay_policy.failed_claim == "repair_required"
        assert unit.fallback_policy.default == "fail_closed"
        assert unit.validate() == []

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "code_unit_id: x\nsymbol: x\nsource_path: x.py\nowner: ci\n"
            "decision_receipts: []\nclaims: []\nassumptions: []\nverifiers: []\n"
            "freshness_sla_hours: 0\n"
            "decay_policy: {failed_claim: report_only, stale_evidence: report_only, unresolved_crux: report_only}\n"
            "fallback_policy: {default: fail_closed, operator_message: ''}\n"
        )
        with pytest.raises(ValueError, match="freshness_sla_hours"):
            load_proof_unit_from_yaml(bad)

    def test_valid_yaml_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "unit.yaml"
        path.write_text(
            "code_unit_id: test.unit.alpha\nsymbol: aragora.epistemic.proof_unit.load_proof_unit\n"
            "source_path: aragora/epistemic/proof_unit.py\nowner: ci\n"
            "decision_receipts: [r.1]\nclaims: [c.1]\nassumptions: [A.]\n"
            "verifiers: [{kind: command, command: 'echo ok'}]\nfreshness_sla_hours: 24\n"
            "decay_policy: {failed_claim: report_only, stale_evidence: report_only, unresolved_crux: report_only}\n"
            "fallback_policy: {default: fail_closed, operator_message: stop}\n"
        )
        assert load_proof_unit_from_yaml(path).validate() == []
