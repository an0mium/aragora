"""Tests for the DIC-22 repair-spec CLI (issue #6033).

Hermetic: no network calls, no file-system side effects beyond tmp_path.
All imports from aragora.epistemic happen inside tested functions so the
module is always importable regardless of optional dependencies.
"""

from __future__ import annotations

import argparse
import json
import os

import pytest

from aragora.cli.commands.dic22_repair import _FLAG, _parse_decay_signal, cmd_repair_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_SIGNAL = {
    "code_unit_id": "proof_first.shift.green_criteria",
    "integrity_score": 0.42,
    "reasons": [
        {
            "kind": "failed_claim",
            "detail": "bc12.benchmark_surface_fresh",
            "claim_id": "bc12.bench",
        },
        {"kind": "unresolved_crux", "detail": "soak policy", "crux_id": "crux.soak"},
    ],
    "recommended_action": "repair_required",
}


def _make_args(
    signal_file: str, kind: str = "report_only", as_json: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(signal_file=signal_file, kind=kind, json=as_json)


# ---------------------------------------------------------------------------
# Flag gate (3 tests)
# ---------------------------------------------------------------------------


def test_flag_off_exits_1_and_names_flag(tmp_path, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 1


def test_flag_truthy_value_1_exits_0(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 0


def test_flag_truthy_value_yes_exits_0(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "yes")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 0


# ---------------------------------------------------------------------------
# Input validation (4 tests)
# ---------------------------------------------------------------------------


def test_missing_signal_file_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    rc = cmd_repair_spec(_make_args(str(tmp_path / "nonexistent.json")))
    assert rc == 1


def test_invalid_json_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "bad.json"
    sig.write_text("{not json}")
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 1


def test_missing_code_unit_id_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    bad = dict(_MINIMAL_SIGNAL)
    bad.pop("code_unit_id")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(bad))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 1


def test_live_swap_kind_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig), kind="live_swap"))
    assert rc == 1


# ---------------------------------------------------------------------------
# Text output (3 tests)
# ---------------------------------------------------------------------------


def test_text_output_contains_spec_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "repair-spec:" in out


def test_text_output_contains_code_unit_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "proof_first.shift.green_criteria" in out


def test_text_output_contains_repair_kind(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "report_only" in out


# ---------------------------------------------------------------------------
# JSON output (3 tests)
# ---------------------------------------------------------------------------


def test_json_output_is_valid_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig), as_json=True))
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, dict)


def test_json_output_has_required_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    cmd_repair_spec(_make_args(str(sig), as_json=True))
    out = capsys.readouterr().out
    parsed = json.loads(out)
    for key in ("spec_id", "code_unit_id", "repair_kind", "provenance_hash"):
        assert key in parsed, f"missing key: {key}"


def test_json_output_report_only_has_empty_provenance_hash(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    cmd_repair_spec(_make_args(str(sig), as_json=True))
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["provenance_hash"] == "", "report_only must have empty provenance_hash"


# ---------------------------------------------------------------------------
# Repair-kind routing (4 tests)
# ---------------------------------------------------------------------------


def test_report_only_exits_0(tmp_path, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    assert cmd_repair_spec(_make_args(str(sig), kind="report_only")) == 0


def test_shadow_candidate_exits_0_when_flag_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    rc = cmd_repair_spec(_make_args(str(sig), kind="shadow_candidate", as_json=True))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["repair_kind"] == "shadow_candidate"


def test_shadow_candidate_has_nonempty_provenance_hash(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    cmd_repair_spec(_make_args(str(sig), kind="shadow_candidate", as_json=True))
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["provenance_hash"]) == 64, (
        "shadow_candidate must have 64-char provenance hash"
    )


def test_pr_candidate_has_nonempty_provenance_hash(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(_FLAG, "1")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(_MINIMAL_SIGNAL))
    cmd_repair_spec(_make_args(str(sig), kind="pr_candidate", as_json=True))
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["provenance_hash"]) == 64, "pr_candidate must have 64-char provenance hash"


# ---------------------------------------------------------------------------
# _parse_decay_signal unit tests (3 tests)
# ---------------------------------------------------------------------------


def test_parse_decay_signal_extracts_claims_and_cruxes():
    sig = _parse_decay_signal(_MINIMAL_SIGNAL)
    assert sig.code_unit_id == "proof_first.shift.green_criteria"
    assert abs(sig.integrity_score - 0.42) < 1e-6
    claim_ids = [r.claim_id for r in sig.reasons if r.claim_id]
    crux_ids = [r.crux_id for r in sig.reasons if r.crux_id]
    assert "bc12.bench" in claim_ids
    assert "crux.soak" in crux_ids


def test_parse_decay_signal_missing_code_unit_id_raises():
    bad = dict(_MINIMAL_SIGNAL)
    bad.pop("code_unit_id")
    with pytest.raises(ValueError, match="code_unit_id"):
        _parse_decay_signal(bad)


def test_parse_decay_signal_clamps_score_to_unit_interval():
    data = dict(_MINIMAL_SIGNAL)
    data["integrity_score"] = 1.5
    sig = _parse_decay_signal(data)
    assert sig.integrity_score == 1.0

    data["integrity_score"] = -0.2
    sig = _parse_decay_signal(data)
    assert sig.integrity_score == 0.0
