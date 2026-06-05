"""Tests for aragora.cli.commands.dic19_proof_units (DIC-19 / #6030).

All tests are hermetic; tmpdir for YAML fixtures.
No subprocess execution, no network, no queue mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import aragora.epistemic.proof_unit_scanner as _scanner

from aragora.cli.commands.dic19_proof_units import _FLAG, cmd_proof_units

_UNIT_A = """\
code_unit_id: proof_first.shift.green_criteria
symbol: scripts.run_proof_first_shift.evaluate_green_shift
source_path: scripts/run_proof_first_shift.py
owner: proof-first-runtime
decision_receipts:
  - decision.bc12.green_shift_criteria
claims:
  - b0.benchmark_truth.complete_current_corpus
  - docs.proof_first_queue_policy.current
assumptions:
  - "Benchmark freshness can be determined."
verifiers:
  - kind: command
    command: "echo ok"
freshness_sla_hours: 24
decay_policy:
  failed_claim: repair_required
  stale_evidence: report_only
  unresolved_crux: report_only
fallback_policy:
  default: fail_closed
  operator_message: "Stop."
linked_crux_ids: []
"""

_UNIT_B = """\
code_unit_id: queue.admission.preflight
symbol: aragora.swarm.preflight.run_preflight
source_path: aragora/swarm/preflight.py
owner: swarm-team
decision_receipts:
  - decision.admission.gate.v1
claims:
  - swarm.admission.preflight.passes
assumptions:
  - "Preflight checks are current."
verifiers:
  - kind: command
    command: "echo ok"
freshness_sla_hours: 12
decay_policy:
  failed_claim: fail_closed
  stale_evidence: report_only
  unresolved_crux: report_only
fallback_policy:
  default: fail_closed
  operator_message: "Admission blocked."
linked_crux_ids:
  - crux.admission.preflight.validity
"""


def _dir(tmp_path: Path, units: list[str]) -> Path:
    d = tmp_path / "proof_units"
    d.mkdir()
    for i, content in enumerate(units):
        (d / f"unit_{i:02d}.yaml").write_text(content, encoding="utf-8")
    return d


def _ns(**kw) -> argparse.Namespace:
    d = {"proof_units_dir": None, "impact_of": None, "multi_hop": False, "json": False}
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture(autouse=True)
def _reset_scan():
    yield
    _scanner.reset_proof_unit_scan()


class TestFlagGating:
    def test_flag_off_exits_1(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        assert cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A])))) == 1

    def test_flag_off_names_flag_in_stderr(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A]))))
        assert _FLAG in capsys.readouterr().err

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_flag_truthy_exits_0(self, monkeypatch, tmp_path, val) -> None:
        monkeypatch.setenv(_FLAG, val)
        assert cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A])))) == 0


class TestMissingDir:
    def test_missing_dir_exits_1(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(_FLAG, "1")
        assert cmd_proof_units(_ns(proof_units_dir=str(tmp_path / "nope"))) == 1

    def test_missing_dir_stderr_names_path(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        bad = str(tmp_path / "nope")
        cmd_proof_units(_ns(proof_units_dir=bad))
        assert bad in capsys.readouterr().err


class TestTextOutput:
    def test_empty_dir_exits_0(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(_FLAG, "1")
        d = tmp_path / "empty"
        d.mkdir()
        assert cmd_proof_units(_ns(proof_units_dir=str(d))) == 0

    def test_unit_count_in_output(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A, _UNIT_B]))))
        assert "2 total" in capsys.readouterr().out

    def test_unit_id_and_claim_in_output(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A]))))
        out = capsys.readouterr().out
        assert "proof_first.shift.green_criteria" in out
        assert "b0.benchmark_truth.complete_current_corpus" in out

    def test_crux_id_shown_when_present(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_B]))))
        assert "crux.admission.preflight.validity" in capsys.readouterr().out


class TestJsonOutput:
    def test_json_parseable_with_generated_at(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A])), json=True))
        data = json.loads(capsys.readouterr().out)
        assert "generated_at" in data

    def test_json_unit_count(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        cmd_proof_units(_ns(proof_units_dir=str(_dir(tmp_path, [_UNIT_A, _UNIT_B])), json=True))
        data = json.loads(capsys.readouterr().out)
        assert data["unit_count"] == 2


class TestImpactOf:
    def test_known_claim_returns_impacted_unit(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        rc = cmd_proof_units(
            _ns(
                proof_units_dir=str(_dir(tmp_path, [_UNIT_A, _UNIT_B])),
                impact_of=["b0.benchmark_truth.complete_current_corpus"],
            )
        )
        assert rc == 0
        assert "proof_first.shift.green_criteria" in capsys.readouterr().out

    def test_unknown_claim_shows_zero_impacted(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        rc = cmd_proof_units(
            _ns(
                proof_units_dir=str(_dir(tmp_path, [_UNIT_A])),
                impact_of=["claim.does.not.exist"],
            )
        )
        assert rc == 0
        assert "0" in capsys.readouterr().out

    def test_impact_json_has_total_and_query_claims(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        claim = "b0.benchmark_truth.complete_current_corpus"
        cmd_proof_units(
            _ns(
                proof_units_dir=str(_dir(tmp_path, [_UNIT_A])),
                impact_of=[claim],
                json=True,
            )
        )
        data = json.loads(capsys.readouterr().out)
        assert data["total"] == 1
        assert claim in data["query_claims"]
        assert data["multi_hop"] is False
