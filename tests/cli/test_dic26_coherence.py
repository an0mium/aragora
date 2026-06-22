"""DIC-26 coherence-scan CLI tests.

Flag: ARAGORA_COHERENCE_MONITOR_ENABLED (default OFF)
Live queue effect: none
Advances: issue #6220 (DIC-26)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands.dic26_coherence import _load_entries, cmd_coherence_scan


def _args(
    input_path: str,
    *,
    json_output: bool = False,
    gap: float = 0.5,
    min_conf: float = 0.3,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.input = input_path
    ns.json = json_output
    ns.contradiction_gap = gap
    ns.min_confidence = min_conf
    return ns


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "beliefs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_disabled_by_default_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ARAGORA_COHERENCE_MONITOR_ENABLED", raising=False)
    assert cmd_coherence_scan(_args(str(_write(tmp_path, [])))) == 1


def test_missing_input_file_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARAGORA_COHERENCE_MONITOR_ENABLED", "1")
    assert cmd_coherence_scan(_args(str(tmp_path / "nonexistent.json"))) == 1


def test_empty_ledger_exits_0_and_reports_coherent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARAGORA_COHERENCE_MONITOR_ENABLED", "1")
    assert cmd_coherence_scan(_args(str(_write(tmp_path, [])))) == 0
    assert "coherent" in capsys.readouterr().out


def test_json_output_is_valid_for_coherent_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARAGORA_COHERENCE_MONITOR_ENABLED", "1")
    entries = [
        {"belief_id": "b1", "subject": "claim.rate_limit", "confidence": 0.9, "status": "pass"}
    ]
    assert cmd_coherence_scan(_args(str(_write(tmp_path, entries)), json_output=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["scanned"] == 1 and data["coherent"] is True and data["issue_count"] == 0


def test_contradiction_flagged_in_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARAGORA_COHERENCE_MONITOR_ENABLED", "1")
    entries = [
        {"belief_id": "b_high", "subject": "claim.x", "confidence": 0.95, "status": "pass"},
        {"belief_id": "b_low", "subject": "claim.x", "confidence": 0.05, "status": "fail"},
    ]
    assert cmd_coherence_scan(_args(str(_write(tmp_path, entries)), json_output=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["contradiction_count"] == 1 and data["coherent"] is False


def test_confidence_rot_flagged_in_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARAGORA_COHERENCE_MONITOR_ENABLED", "1")
    entries = [
        {"belief_id": "b_rotten", "subject": "claim.stale", "confidence": 0.1, "status": "stale"}
    ]
    assert (
        cmd_coherence_scan(_args(str(_write(tmp_path, entries)), json_output=True, min_conf=0.3))
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["confidence_rot_count"] == 1


def test_load_entries_skips_bad_confidence_type(tmp_path: Path) -> None:
    raw = [
        {"belief_id": "good", "subject": "s", "confidence": 0.8},
        {"belief_id": "bad_conf", "subject": "s", "confidence": "not-a-float"},
    ]
    p = tmp_path / "b.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    entries = _load_entries(p)
    assert len(entries) == 1 and entries[0].belief_id == "good"


def test_load_entries_skips_missing_belief_id(tmp_path: Path) -> None:
    raw = [
        {"belief_id": "ok", "subject": "s", "confidence": 0.7},
        {"subject": "no_id", "confidence": 0.5},
    ]
    p = tmp_path / "b.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    entries = _load_entries(p)
    assert len(entries) == 1 and entries[0].belief_id == "ok"


def test_load_entries_accepts_single_object(tmp_path: Path) -> None:
    raw = {"belief_id": "singleton", "subject": "s", "confidence": 0.6}
    p = tmp_path / "b.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    entries = _load_entries(p)
    assert len(entries) == 1 and entries[0].belief_id == "singleton"
