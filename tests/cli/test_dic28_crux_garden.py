"""Tests for aragora crux-garden CLI (DIC-28 / #6222).

No network, no subprocess, no queue mutation.
All receipt inputs are synthetic dicts.
"""

from __future__ import annotations

import sys
import types

# yaml stub — pyyaml is absent in the hermetic uv-tool pytest venv used for
# vision-layer CI.  Provides the subset of yaml needed by transitive imports
# from aragora.epistemic (only yaml.safe_load is called at import time).
if "yaml" not in sys.modules:
    _yaml_stub = types.ModuleType("yaml")
    _yaml_stub.safe_load = lambda stream: {}  # type: ignore[attr-defined]
    _yaml_stub.dump = lambda data, **kw: ""  # type: ignore[attr-defined]
    sys.modules["yaml"] = _yaml_stub

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands.dic28_crux_garden import (
    _flag_enabled,
    _load_receipts,
    _parse_receipt,
    cmd_crux_garden,
)

_FLAG = "ARAGORA_CRUX_GARDENING_ENABLED"

_RECEIPT: dict = {
    "receipt_id": "rcpt_abc123",
    "debate_id": "debate_xyz",
    "question": "Should we expand the B2 guard now?",
    "cruxes": [
        {
            "crux_id": "crux.soak.equivalence",
            "statement": "One soak may equal three.",
            "load_bearing_score": 0.82,
            "uncertainty_score": 0.40,
            "contesting_agents": ["claude", "codex"],
            "affected_claims": ["claim.b0.fresh"],
            "resolution_impact": 0.9,
        }
    ],
    "convergence_barrier": 0.73,
    "counterfactuals": [],
    "agents": ["claude", "codex"],
    "rounds": 3,
    "metadata": {"mode": "crux_finder"},
    "checksum": "deadbeef" * 8,
}


def _ns(
    *,
    input_path: str,
    json_out: bool = False,
    write: bool = False,
    output_dir: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(input=input_path, json=json_out, write=write, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Flag guard
# ---------------------------------------------------------------------------


def test_flag_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert _flag_enabled() is False


def test_disabled_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    f = tmp_path / "r.json"
    f.write_text("[]")
    assert cmd_crux_garden(_ns(input_path=str(f))) == 1
    assert _FLAG in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_missing_input_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_crux_garden(_ns(input_path=str(tmp_path / "nope.json"))) == 1
    assert "not found" in capsys.readouterr().err


def test_invalid_json_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "bad.json"
    f.write_text("[{invalid}")
    assert cmd_crux_garden(_ns(input_path=str(f))) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_jsonl_bad_line_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps(_RECEIPT) + "\n{bad}\n")
    assert cmd_crux_garden(_ns(input_path=str(f))) == 1
    assert "line 2" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------


def test_json_array_input_exits_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    assert cmd_crux_garden(_ns(input_path=str(f))) == 0


def test_jsonl_input_exits_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps(_RECEIPT) + "\n")
    assert cmd_crux_garden(_ns(input_path=str(f))) == 0


def test_json_output_parseable_with_schema_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    assert cmd_crux_garden(_ns(input_path=str(f), json_out=True)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert "summary" in report
    assert "generated_at" in report


def test_summary_has_gardening_status_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    cmd_crux_garden(_ns(input_path=str(f), json_out=True))
    report = json.loads(capsys.readouterr().out)
    for key in ("healthy", "stale_evidence", "new_contradiction", "insufficient_evidence"):
        assert key in report["summary"], f"missing summary key: {key}"


def test_parse_receipt_crux_fields(tmp_path: Path) -> None:
    receipt = _parse_receipt(_RECEIPT)
    assert len(receipt.cruxes) == 1
    crux = receipt.cruxes[0]
    assert crux.crux_id == "crux.soak.equivalence"
    assert crux.load_bearing_score == pytest.approx(0.82)
    assert "claim.b0.fresh" in crux.affected_claims


@pytest.mark.parametrize(
    "content",
    [
        "[1]",  # array with non-object scalar
        "[null]",  # array with null entry
        '["string"]',  # array with string entry
    ],
)
def test_non_object_receipt_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    content: str,
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(content)
    assert cmd_crux_garden(_ns(input_path=str(f))) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_jsonl_non_object_line_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.jsonl"
    f.write_text("1\n")
    assert cmd_crux_garden(_ns(input_path=str(f))) == 1
    assert "line 1" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --write flag: persist report to output directory
# ---------------------------------------------------------------------------


def test_write_flag_creates_report_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    out_dir = tmp_path / "reports"
    rc = cmd_crux_garden(_ns(input_path=str(f), write=True, output_dir=str(out_dir)))
    assert rc == 0
    files = list(out_dir.glob("gardening_report_*.json"))
    assert len(files) == 1


def test_write_flag_report_json_is_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    out_dir = tmp_path / "reports"
    cmd_crux_garden(_ns(input_path=str(f), write=True, output_dir=str(out_dir)))
    files = list(out_dir.glob("gardening_report_*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "generated_at" in data
    assert "resolved_results" in data
    assert "outstanding_results" in data
    assert "schema_version" in data


def test_write_flag_creates_output_dir_if_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    rc = cmd_crux_garden(_ns(input_path=str(f), write=True, output_dir=str(nested)))
    assert rc == 0
    assert nested.exists()
    assert any(nested.glob("gardening_report_*.json"))


def test_write_flag_off_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    f = tmp_path / "r.json"
    f.write_text(json.dumps([_RECEIPT]))
    out_dir = tmp_path / "reports"
    rc = cmd_crux_garden(_ns(input_path=str(f), write=False, output_dir=str(out_dir)))
    assert rc == 0
    assert not out_dir.exists()
