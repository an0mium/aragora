"""Tests for aragora.epistemic.proof_unit (DIC-19 / #6030).

Pure schema/validation — no network, no subprocess, no queue mutation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import aragora.epistemic.proof_unit_scanner as _proof_unit_scanner
from aragora.epistemic.proof_unit import (
    DecayPolicy,
    FallbackPolicy,
    ProofCarryingCodeUnit,
    enable_proof_unit_scan,
    load_proof_unit,
    load_proof_unit_from_yaml,
    load_proof_units_from_dir,
    proof_unit_scan_enabled,
    reset_proof_unit_scan,
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


_VALID_YAML = (
    "code_unit_id: test.scan.alpha\n"
    "symbol: aragora.epistemic.proof_unit.load_proof_unit\n"
    "source_path: aragora/epistemic/proof_unit.py\n"
    "owner: ci\n"
    "decision_receipts: [r.1]\n"
    "claims: [c.1]\n"
    "assumptions: [A.]\n"
    "verifiers: [{kind: command, command: 'echo ok'}]\n"
    "freshness_sla_hours: 24\n"
    "decay_policy: {failed_claim: report_only, stale_evidence: report_only, unresolved_crux: report_only}\n"
    "fallback_policy: {default: fail_closed, operator_message: ''}\n"
)


@pytest.fixture(autouse=True)
def _reset_scan_override() -> pytest.IterableFixture:  # type: ignore[type-arg]
    """Ensure module-level scan override is cleared around every test."""
    reset_proof_unit_scan()
    yield
    reset_proof_unit_scan()


class TestScanFlag:
    def test_scan_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_PROOF_UNIT_SCAN_ENABLED", raising=False)
        assert not proof_unit_scan_enabled()

    def test_scan_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_PROOF_UNIT_SCAN_ENABLED", "1")
        assert proof_unit_scan_enabled()

    def test_scan_enabled_via_true_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_PROOF_UNIT_SCAN_ENABLED", "true")
        assert proof_unit_scan_enabled()

    def test_enable_helper_sets_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_PROOF_UNIT_SCAN_ENABLED", raising=False)
        enable_proof_unit_scan()
        assert proof_unit_scan_enabled()
        assert _proof_unit_scanner._scan_enabled_override is True  # noqa: SLF001

    def test_enable_does_not_mutate_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_PROOF_UNIT_SCAN_ENABLED", raising=False)
        before = dict(os.environ)
        enable_proof_unit_scan()
        assert os.environ == before

    def test_reset_clears_override(self) -> None:
        enable_proof_unit_scan()
        reset_proof_unit_scan()
        assert _proof_unit_scanner._scan_enabled_override is None  # noqa: SLF001
        assert not proof_unit_scan_enabled()

    def test_override_takes_priority_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_PROOF_UNIT_SCAN_ENABLED", raising=False)
        enable_proof_unit_scan()
        assert proof_unit_scan_enabled()

    def test_load_from_dir_off_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "unit.yaml").write_text(_VALID_YAML)
        assert load_proof_units_from_dir(tmp_path) == []

    def test_load_from_dir_enabled_parses_valid(self, tmp_path: Path) -> None:
        enable_proof_unit_scan()
        (tmp_path / "unit.yaml").write_text(_VALID_YAML)
        units = load_proof_units_from_dir(tmp_path)
        assert len(units) == 1
        assert units[0].code_unit_id == "test.scan.alpha"

    def test_load_from_dir_logs_validation_errors(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        enable_proof_unit_scan()
        (tmp_path / "bad.yaml").write_text(
            "code_unit_id: x\nfreshness_sla_hours: 0\n"
            "symbol: x\nsource_path: x.py\nowner: ci\n"
            "decision_receipts: []\nclaims: []\nassumptions: []\nverifiers: []\n"
            "decay_policy: {failed_claim: report_only, stale_evidence: report_only, unresolved_crux: report_only}\n"
            "fallback_policy: {default: fail_closed, operator_message: ''}\n"
        )
        with caplog.at_level("WARNING", logger="aragora.epistemic.proof_unit_scanner"):
            assert load_proof_units_from_dir(tmp_path) == []
        assert "skipping invalid proof unit" in caplog.text

    def test_load_from_dir_logs_malformed_with_traceback(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        enable_proof_unit_scan()
        (tmp_path / "bad.yaml").write_text("- not-a-mapping\n")
        with caplog.at_level("WARNING", logger="aragora.epistemic.proof_unit_scanner"):
            assert load_proof_units_from_dir(tmp_path) == []
        assert "skipping malformed proof unit" in caplog.text
        assert any(record.exc_info for record in caplog.records)

    def test_load_from_dir_returns_sorted_order(self, tmp_path: Path) -> None:
        enable_proof_unit_scan()
        for name, unit_id in [("b_unit.yaml", "test.scan.b"), ("a_unit.yaml", "test.scan.a")]:
            (tmp_path / name).write_text(_VALID_YAML.replace("test.scan.alpha", unit_id))
        units = load_proof_units_from_dir(tmp_path)
        assert [u.code_unit_id for u in units] == ["test.scan.a", "test.scan.b"]

    def test_load_from_dir_proof_units_dir_enabled(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(PROOF_UNITS_DIR)
        assert len(units) >= 1
        ids = [u.code_unit_id for u in units]
        assert "proof_first.shift.green_criteria" in ids
