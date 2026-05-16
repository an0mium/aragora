"""Tests for aragora.cli.commands.dic24_genealogy (DIC-24 / #6218).

Run with:
    pytest tests/cli/test_dic24_genealogy.py --noconftest

All tests are hermetic; tmpdir for JSONL files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands.dic24_genealogy import (
    _FLAG,
    cmd_genealogy_show,
)

_A = {
    "code_unit_id": "proof_first.shift",
    "entry_kind": "decay_signal",
    "entry_id": "decay-001",
    "checksum": "aabbcc",
    "timestamp": "2026-04-25T10:00:00Z",
}
_B = {
    "code_unit_id": "proof_first.shift",
    "entry_kind": "repair_proposal",
    "entry_id": "repair-001",
    "checksum": "ddeeff",
    "timestamp": "2026-04-26T12:00:00Z",
}


def _store(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "gen.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    return p


def _ns(**kw) -> argparse.Namespace:
    d = {"store_file": ".aragora_genealogy.jsonl", "json": False}
    d.update(kw)
    return argparse.Namespace(**d)


# -- Flag gating --


class TestFlagGating:
    def test_exits_1_when_flag_off(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        assert (
            cmd_genealogy_show(_ns(code_unit_id="x", store_file=str(_store(tmp_path, [_A])))) == 1
        )

    def test_error_message_names_flag(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.delenv(_FLAG, raising=False)
        cmd_genealogy_show(_ns(code_unit_id="x", store_file=str(_store(tmp_path, [_A]))))
        assert _FLAG in capsys.readouterr().err

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_flag_truthy_values_accepted(self, monkeypatch, tmp_path, val: str) -> None:
        monkeypatch.setenv(_FLAG, val)
        args = _ns(code_unit_id="proof_first.shift", store_file=str(_store(tmp_path, [_A])))
        assert cmd_genealogy_show(args) == 0


# -- show: text output --


class TestShowText:
    def test_shows_code_unit_id_and_count(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(code_unit_id="proof_first.shift", store_file=str(_store(tmp_path, [_A, _B])))
        cmd_genealogy_show(args)
        out = capsys.readouterr().out
        assert "proof_first.shift" in out and "2" in out

    def test_unknown_unit_shows_no_entries(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(code_unit_id="nonexistent", store_file=str(_store(tmp_path, [_A])))
        assert cmd_genealogy_show(args) == 0
        assert "no entries" in capsys.readouterr().out

    def test_missing_store_file_ok(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(code_unit_id="x", store_file=str(tmp_path / "missing.jsonl"))
        assert cmd_genealogy_show(args) == 0
        assert "no entries" in capsys.readouterr().out

    def test_entry_kind_and_id_appear_in_output(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(code_unit_id="proof_first.shift", store_file=str(_store(tmp_path, [_A])))
        cmd_genealogy_show(args)
        out = capsys.readouterr().out
        assert "decay_signal" in out and "decay-001" in out


# -- show: JSON output --


class TestShowJson:
    def test_json_output_has_correct_fields(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(
            code_unit_id="proof_first.shift", store_file=str(_store(tmp_path, [_A, _B])), json=True
        )
        cmd_genealogy_show(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["code_unit_id"] == "proof_first.shift"
        assert payload["entry_count"] == 2
        assert len(payload["chain_checksum"]) == 64

    def test_entries_sorted_oldest_first(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(
            code_unit_id="proof_first.shift", store_file=str(_store(tmp_path, [_B, _A])), json=True
        )
        cmd_genealogy_show(args)
        ts = [e["timestamp"] for e in json.loads(capsys.readouterr().out)["entries"]]
        assert ts == sorted(ts)

    def test_json_unknown_unit_has_zero_entries(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        args = _ns(code_unit_id="no.such.unit", store_file=str(_store(tmp_path, [_A])), json=True)
        cmd_genealogy_show(args)
        assert json.loads(capsys.readouterr().out)["entry_count"] == 0


# -- JSONL parsing resilience --


class TestJsonlParsing:
    def test_malformed_line_skipped(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(_FLAG, "1")
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps(_A) + "\nnot-json\n" + json.dumps(_B), encoding="utf-8")
        args = _ns(code_unit_id="proof_first.shift", store_file=str(p), json=True)
        import io
        import sys

        old, sys.stdout = sys.stdout, io.StringIO()
        rc = cmd_genealogy_show(args)
        out, sys.stdout = sys.stdout.getvalue(), old
        assert rc == 0 and json.loads(out)["entry_count"] == 2

    def test_blank_lines_ignored(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setenv(_FLAG, "1")
        p = tmp_path / "blanks.jsonl"
        p.write_text("\n\n" + json.dumps(_A) + "\n\n", encoding="utf-8")
        args = _ns(code_unit_id="proof_first.shift", store_file=str(p), json=True)
        cmd_genealogy_show(args)
        assert json.loads(capsys.readouterr().out)["entry_count"] == 1
