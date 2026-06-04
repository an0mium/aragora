"""Tests for aragora.cli.commands.dic21_quarantine (DIC-21 / #6032).

All tests are hermetic; tmp_path for JSON signal fixtures.
Live queue effect: none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# aragora.epistemic.__init__ eagerly imports claim_verifier which does
# ``import yaml`` at module level.  Stub it before the first aragora.epistemic
# import so the package initialises cleanly in environments where pyyaml is
# absent.  The quarantine_policy module under test is pure Python; no YAML
# is parsed.
if "yaml" not in sys.modules:
    sys.modules["yaml"] = MagicMock()  # type: ignore[assignment]

from aragora.cli.commands.dic21_quarantine import _FLAG, cmd_quarantine_eval  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FRESH: dict = {
    "code_unit_id": "test.unit.alpha",
    "integrity_score": 0.9,
    "reasons": [],
    "recommended_action": "report_only",
}

_DECAYED: dict = {
    "code_unit_id": "test.unit.beta",
    "integrity_score": 0.35,
    "reasons": [{"kind": "failed_claim", "detail": "b0.truth.rate failed"}],
    "recommended_action": "repair_required",
}


def _ns(
    signal: str,
    code_unit_class: str = "default",
    request_live_swap: bool = False,
    json_out: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        signal=signal,
        code_unit_class=code_unit_class,
        request_live_swap=request_live_swap,
        json=json_out,
    )


def _sig(tmp_path: Path, data: dict, name: str = "sig.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------


def test_flag_off_exits_1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH)))) == 1


def test_flag_off_names_flag_in_stderr(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH))))
    assert _FLAG in capsys.readouterr().err


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_flag_truthy_values_exit_0(monkeypatch, tmp_path: Path, val: str) -> None:
    monkeypatch.setenv(_FLAG, val)
    assert cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH)))) == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_signal_file_exits_1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_quarantine_eval(_ns(str(tmp_path / "absent.json"))) == 1


def test_bad_json_exits_1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert cmd_quarantine_eval(_ns(str(p))) == 1


def test_missing_code_unit_id_exits_1(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_quarantine_eval(_ns(str(_sig(tmp_path, {"integrity_score": 0.9})))) == 1


# ---------------------------------------------------------------------------
# Output — text
# ---------------------------------------------------------------------------


def test_text_output_contains_unit_id(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH))))
    assert "test.unit.alpha" in capsys.readouterr().out


def test_text_output_shows_action(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH))))
    assert "report_only" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Output — JSON
# ---------------------------------------------------------------------------


def test_json_output_is_valid(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH)), json_out=True))
    d = json.loads(capsys.readouterr().out)
    assert {"code_unit_id", "policy_action", "integrity_score"} <= d.keys()


def test_non_report_only_carries_provenance_hash(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _DECAYED)), json_out=True))
    assert len(json.loads(capsys.readouterr().out)["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# Policy routing
# ---------------------------------------------------------------------------


def test_high_integrity_is_report_only(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH)), json_out=True))
    assert json.loads(capsys.readouterr().out)["policy_action"] == "report_only"


def test_low_integrity_triggers_fail_closed(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    sig = {**_FRESH, "integrity_score": 0.1, "code_unit_id": "test.unit.low"}
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, sig)), json_out=True))
    result = json.loads(capsys.readouterr().out)
    assert result["policy_action"] == "fail_closed"
    assert result["fail_closed"] is True


def test_live_dispatch_class_uses_higher_threshold(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv(_FLAG, "1")
    # integrity 0.55 is > default threshold (0.4) but < live_dispatch threshold (0.6)
    sig = {**_FRESH, "integrity_score": 0.55, "code_unit_id": "test.unit.ld"}
    cmd_quarantine_eval(
        _ns(str(_sig(tmp_path, sig)), code_unit_class="live_dispatch", json_out=True)
    )
    result = json.loads(capsys.readouterr().out)
    assert result["policy_action"] == "fail_closed"
    assert result["fail_closed"] is True


def test_live_swap_request_without_allowlist_is_blocked(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_quarantine_eval(_ns(str(_sig(tmp_path, _FRESH)), request_live_swap=True, json_out=True))
    assert json.loads(capsys.readouterr().out)["live_swap_blocked"] is True
