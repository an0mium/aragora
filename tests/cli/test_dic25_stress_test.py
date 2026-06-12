"""Tests for aragora.cli.commands.dic25_stress_test (DIC-25 / #6219).

Hermetic: no network, queue, or database access.

Note: the `aragora.epistemic` package __init__ imports `claim_verifier` which
requires pyyaml. Stub it here so the hermetic uv-tool pytest venv (no pyyaml
installed) can import the package cleanly. The same pattern is used by DIC-21
and DIC-23 CLI tests.
"""

from __future__ import annotations

import sys
import types

# Install yaml stub before any aragora.epistemic import is triggered.
if "yaml" not in sys.modules:
    _yaml = types.ModuleType("yaml")
    _yaml.safe_load = lambda s: None  # type: ignore[attr-defined]
    sys.modules["yaml"] = _yaml

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands.dic25_stress_test import _FLAG, cmd_stress_test

_CATALOG_1 = [
    {
        "perturbation_id": "p1",
        "kind": "cve_drop",
        "description": "OpenSSL CVE drop — global scope",
        "simulated_impact": 0.4,
        "affected_claim_ids": [],
        "affected_proof_unit_ids": [],
    }
]
_UNITS_1: dict[str, float] = {
    "proof_first.shift.green_criteria": 0.9,
    "benchmark.truth.publication": 0.8,
}


def _ns(catalog: str, units: str, json_out: bool = False) -> argparse.Namespace:
    return argparse.Namespace(catalog=catalog, units=units, json=json_out)


@pytest.fixture()
def catalog_file(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(_CATALOG_1), encoding="utf-8")
    return p


@pytest.fixture()
def units_file(tmp_path: Path) -> Path:
    p = tmp_path / "units.json"
    p.write_text(json.dumps(_UNITS_1), encoding="utf-8")
    return p


# ── Flag gating ──────────────────────────────────────────────────────────────


def test_flag_off_exits_1(
    monkeypatch: pytest.MonkeyPatch, catalog_file: Path, units_file: Path
) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    assert cmd_stress_test(_ns(str(catalog_file), str(units_file))) == 1


def test_flag_off_names_flag_in_stderr(
    monkeypatch: pytest.MonkeyPatch,
    catalog_file: Path,
    units_file: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    cmd_stress_test(_ns(str(catalog_file), str(units_file)))
    assert _FLAG in capsys.readouterr().err


@pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
def test_flag_truthy_values_exit_0(
    monkeypatch: pytest.MonkeyPatch, catalog_file: Path, units_file: Path, val: str
) -> None:
    monkeypatch.setenv(_FLAG, val)
    assert cmd_stress_test(_ns(str(catalog_file), str(units_file))) == 0


# ── File validation ───────────────────────────────────────────────────────────


def test_missing_catalog_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, units_file: Path
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_stress_test(_ns(str(tmp_path / "missing.json"), str(units_file))) == 1


def test_missing_units_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, catalog_file: Path
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    assert cmd_stress_test(_ns(str(catalog_file), str(tmp_path / "missing.json"))) == 1


# ── Text output ───────────────────────────────────────────────────────────────


def test_text_output_mentions_counts(
    monkeypatch: pytest.MonkeyPatch,
    catalog_file: Path,
    units_file: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_stress_test(_ns(str(catalog_file), str(units_file)))
    out = capsys.readouterr().out
    assert "1 perturbation" in out
    assert "2 unit" in out


def test_high_fragility_appears_in_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, units_file: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    catalog = tmp_path / "catalog_high.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "perturbation_id": "p_high",
                    "kind": "dependency_drop",
                    "description": "Critical dep dropped",
                    "simulated_impact": 0.7,
                    "affected_claim_ids": [],
                    "affected_proof_unit_ids": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert cmd_stress_test(_ns(str(catalog), str(units_file))) == 0
    assert "High-fragility" in capsys.readouterr().out


def test_no_high_fragility_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, units_file: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    catalog = tmp_path / "catalog_low.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "perturbation_id": "p_low",
                    "kind": "api_rate_limit_shift",
                    "description": "Minor rate limit shift",
                    "simulated_impact": 0.05,
                    "affected_claim_ids": [],
                    "affected_proof_unit_ids": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert cmd_stress_test(_ns(str(catalog), str(units_file))) == 0
    assert "No high-fragility" in capsys.readouterr().out


# ── JSON output ───────────────────────────────────────────────────────────────


def test_json_output_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    catalog_file: Path,
    units_file: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    rc = cmd_stress_test(_ns(str(catalog_file), str(units_file), json_out=True))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["perturbations_tested"] == 1
    assert data["proof_units_probed"] == 2
    assert len(data["reports"]) == 2


def test_json_report_fields(
    monkeypatch: pytest.MonkeyPatch,
    catalog_file: Path,
    units_file: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_stress_test(_ns(str(catalog_file), str(units_file), json_out=True))
    rep = json.loads(capsys.readouterr().out)["reports"][0]
    assert "proof_unit_id" in rep
    assert "perturbation_id" in rep
    assert "fragility_delta" in rep
    assert "recommended_action" in rep
    assert "baseline_integrity" in rep


# ── Queue governance ──────────────────────────────────────────────────────────


def test_boss_ready_never_in_output(
    monkeypatch: pytest.MonkeyPatch,
    catalog_file: Path,
    units_file: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    cmd_stress_test(_ns(str(catalog_file), str(units_file), json_out=True))
    raw = capsys.readouterr().out
    assert "boss-ready" not in raw
    assert "boss_ready" not in raw


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_catalog_returns_zero_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, units_file: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    catalog = tmp_path / "empty.json"
    catalog.write_text("[]", encoding="utf-8")
    rc = cmd_stress_test(_ns(str(catalog), str(units_file), json_out=True))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["perturbations_tested"] == 0
    assert data["reports"] == []


def test_scoped_perturbation_skips_out_of_scope_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, units_file: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv(_FLAG, "1")
    catalog = tmp_path / "scoped.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "perturbation_id": "p_scoped",
                    "kind": "corpus_revision",
                    "description": "Scoped to a unit not in units file",
                    "simulated_impact": 0.8,
                    "affected_claim_ids": [],
                    "affected_proof_unit_ids": ["nonexistent.unit"],
                }
            ]
        ),
        encoding="utf-8",
    )
    rc = cmd_stress_test(_ns(str(catalog), str(units_file), json_out=True))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert all(r["fragility_delta"] == 0.0 for r in data["reports"])
