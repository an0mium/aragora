"""CLI tests for DIC-23 ``aragora dialectical-loop`` command.

Flag: ARAGORA_DIALECTICAL_RUNTIME_ENABLED (default OFF)
Live queue effect: none
Advances: issue #6217 (DIC-23)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands.dic23_runtime_loop import _load_signal, cmd_dialectical_loop

_FLAG = "ARAGORA_DIALECTICAL_RUNTIME_ENABLED"


def _args(signal_path: str, *, json_output: bool = False, unit_class: str = "default", repair: bool = False) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.signal = signal_path
    ns.json = json_output
    ns.unit_class = unit_class
    ns.repair = repair
    return ns


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "signal.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _sig(unit_id: str = "u.test", integrity: float = 0.8, action: str = "report_only", reasons: list | None = None) -> dict:
    return {"code_unit_id": unit_id, "integrity_score": integrity, "recommended_action": action, "reasons": reasons or []}


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------

def test_flag_off_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert cmd_dialectical_loop(_args(str(_write(tmp_path, _sig())))) == 1


def test_flag_off_names_flag_in_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    cmd_dialectical_loop(_args(str(_write(tmp_path, _sig()))))
    assert _FLAG in capsys.readouterr().err


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_flag_truthy_values_exit_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, val: str) -> None:
    monkeypatch.setenv(_FLAG, val)
    assert cmd_dialectical_loop(_args(str(_write(tmp_path, _sig())))) == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_signal_file_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_dialectical_loop(_args(str(tmp_path / "absent.json"))) == 1


def test_bad_json_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert cmd_dialectical_loop(_args(str(p))) == 1


def test_missing_code_unit_id_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_dialectical_loop(_args(str(_write(tmp_path, {"integrity_score": 0.9})))) == 1


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------

def test_text_output_contains_event_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_dialectical_loop(_args(str(_write(tmp_path, _sig())))) == 0
    assert "drt_" in capsys.readouterr().out


def test_json_output_is_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_dialectical_loop(_args(str(_write(tmp_path, _sig(unit_id="u.abc"))), json_output=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["code_unit_id"] == "u.abc"
    assert "event_id" in data and "quarantine_action" in data


def test_low_integrity_triggers_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_dialectical_loop(_args(str(_write(tmp_path, _sig(integrity=0.1))), json_output=True))
    assert json.loads(capsys.readouterr().out)["quarantine_action"] == "fail_closed"


def test_live_dispatch_raises_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(_FLAG, "1")
    # live_dispatch fail_closed_threshold=0.6; integrity 0.55 < 0.6 → fail_closed
    cmd_dialectical_loop(_args(str(_write(tmp_path, _sig(integrity=0.55))), json_output=True, unit_class="live_dispatch"))
    assert json.loads(capsys.readouterr().out)["quarantine_action"] == "fail_closed"


# ---------------------------------------------------------------------------
# _load_signal unit tests
# ---------------------------------------------------------------------------

def test_load_signal_parses_reasons_and_defaults(tmp_path: Path) -> None:
    raw = {
        "code_unit_id": "u.x",
        "integrity_score": 0.7,
        "recommended_action": "degrade",
        "reasons": [
            {"kind": "failed_claim", "detail": "foo failed", "claim_id": "c.foo"},
            {"kind": "stale_evidence", "detail": "old"},
        ],
    }
    p = tmp_path / "s.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    sig = _load_signal(p)
    assert sig.code_unit_id == "u.x"
    assert sig.recommended_action == "degrade"
    assert len(sig.reasons) == 2
    assert sig.reasons[0].claim_id == "c.foo"
    assert sig.reasons[1].crux_id == ""


def test_load_signal_defaults_recommended_action(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"code_unit_id": "u.y", "integrity_score": 1.0}), encoding="utf-8")
    sig = _load_signal(p)
    assert sig.recommended_action == "report_only" and sig.reasons == []
