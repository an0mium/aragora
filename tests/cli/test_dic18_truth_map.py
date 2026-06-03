"""Unit tests for the ``aragora truth-map`` CLI command (DIC-18 / #6028)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aragora.cli.commands.dic18_truth_map import _flag_enabled, cmd_truth_map

_FLAG = "ARAGORA_TRUTH_MAP_ENABLED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path: Path, claims: list[dict] | None = None) -> Path:
    """Write a minimal DIC-13 YAML manifest to *tmp_path* and return its path."""
    manifest = {
        "schema_version": 1,
        "manifest_id": "test_manifest",
        "claims": claims or [],
    }
    p = tmp_path / "test_claims.yaml"
    p.write_text(yaml.dump(manifest))
    return p


def _minimal_claim(claim_id: str = "test.claim.one") -> dict:
    return {
        "claim_id": claim_id,
        "statement": "Test claim.",
        "owner": "test-suite",
        "scope": "repo",
        "confidence": "high",
        "freshness_sla_hours": 24,
        "evidence": [{"note": "no file path needed"}],
        "verification": {"kind": "command", "command": "echo ok"},
        "failure": {"severity": "info", "allowed_action": "report_only"},
        "receipts": [{"type": "test_run"}],
    }


def _make_args(claims_dir: str, as_json: bool = False):
    """Construct a minimal argparse.Namespace for cmd_truth_map."""
    import argparse

    return argparse.Namespace(claims_dir=claims_dir, json=as_json)


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------


def test_flag_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert not _flag_enabled()


def test_disabled_exits_1_and_prints_flag_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    _make_manifest(tmp_path)
    rc = cmd_truth_map(_make_args(str(tmp_path)))
    assert rc == 1
    assert _FLAG in capsys.readouterr().err


def test_flag_enabled_on_string_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "true")
    assert _flag_enabled()


# ---------------------------------------------------------------------------
# Missing / empty directory
# ---------------------------------------------------------------------------


def test_missing_claims_dir_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    rc = cmd_truth_map(_make_args(str(tmp_path / "does_not_exist")))
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_empty_claims_dir_exits_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    rc = cmd_truth_map(_make_args(str(tmp_path)))
    assert rc == 0


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_json_output_is_parseable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    _make_manifest(tmp_path, [_minimal_claim()])
    rc = cmd_truth_map(_make_args(str(tmp_path), as_json=True))
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "summary" in data
    assert data["summary"]["total_claims"] == 1


def test_json_output_has_generated_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    _make_manifest(tmp_path, [_minimal_claim()])
    cmd_truth_map(_make_args(str(tmp_path), as_json=True))
    data = json.loads(capsys.readouterr().out)
    assert "generated_at" in data


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------


def test_text_output_contains_claim_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    _make_manifest(tmp_path, [_minimal_claim("my.test.claim")])
    rc = cmd_truth_map(_make_args(str(tmp_path)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "my.test.claim" in out


def test_text_output_contains_summary_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    _make_manifest(tmp_path, [_minimal_claim()])
    cmd_truth_map(_make_args(str(tmp_path)))
    out = capsys.readouterr().out
    assert "total" in out


# ---------------------------------------------------------------------------
# Exit-code semantics
# ---------------------------------------------------------------------------


def test_two_passing_claims_exits_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_FLAG, "1")
    _make_manifest(tmp_path, [_minimal_claim("a.one"), _minimal_claim("a.two")])
    rc = cmd_truth_map(_make_args(str(tmp_path)))
    assert rc == 0


def test_multiple_manifests_aggregated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    for i in range(3):
        m = {
            "schema_version": 1,
            "manifest_id": f"manifest_{i}",
            "claims": [_minimal_claim(f"claim.m{i}")],
        }
        (tmp_path / f"m{i}.yaml").write_text(yaml.dump(m))
    cmd_truth_map(_make_args(str(tmp_path), as_json=True))
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["total_claims"] == 3
