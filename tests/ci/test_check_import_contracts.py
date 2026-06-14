"""Unit tests for scripts/ci/check_import_contracts.py.

These exercise the pure baseline/diff/filter logic and the ``main`` CLI wiring
WITHOUT building the real grimp graph: ``compute_current_violations`` is
monkeypatched so the tests stay fast and deterministic. The end-to-end behavior
against the real ``aragora`` package (green on clean tree, tamper detection,
shrink-only) is covered by the VAL-P0-002/003 acceptance checks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_PATH = REPO_ROOT / "scripts" / "ci" / "check_import_contracts.py"
_REAL_CONFIG = REPO_ROOT / ".importlinter"

_spec = importlib.util.spec_from_file_location("check_import_contracts", _CHECKER_PATH)
cic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cic)

CONTRACT = "aragora-layers"


def _write_baseline(path: Path, violations: set[str], contract: str = CONTRACT) -> None:
    path.write_text(
        json.dumps({"contracts": {contract: {"violations": sorted(violations)}}}),
        encoding="utf-8",
    )


# --- config parsing ---------------------------------------------------------


def test_parse_layer_membership_real_config():
    membership = cic.parse_layer_membership(_REAL_CONFIG)
    assert membership["aragora.server"] == "interface"
    assert membership["aragora.debate"] == "domain"
    assert membership["aragora.storage"] == "infrastructure"
    assert membership["aragora.config"] == "foundation"
    # tail keys resolve too
    assert membership["config"] == "foundation"


def test_layer_of_full_and_tail():
    membership = {"aragora.config": "foundation", "config": "foundation"}
    assert cic.layer_of("aragora.config", membership) == "foundation"
    assert cic.layer_of("aragora.unknown", membership) is None


def test_parse_layers_arg_valid_and_invalid():
    assert cic._parse_layers_arg("foundation,infrastructure") == {
        "foundation",
        "infrastructure",
    }
    with pytest.raises(cic.CheckerError):
        cic._parse_layers_arg("foundation,bogus")


# --- diff / filter ----------------------------------------------------------


def test_diff_against_baseline_new_and_resolved():
    current = {CONTRACT: {"a -> b", "c -> d"}}
    baseline = {CONTRACT: {"a -> b", "e -> f"}}
    new, resolved = cic.diff_against_baseline(current, baseline)
    assert new[CONTRACT] == {"c -> d"}
    assert resolved[CONTRACT] == {"e -> f"}


def test_filter_by_layers_keeps_only_selected_importer_layer():
    membership = {
        "aragora.config": "foundation",
        "aragora.storage": "infrastructure",
    }
    violations = {"aragora.config -> aragora.server", "aragora.storage -> aragora.server"}
    assert cic.filter_by_layers(violations, {"foundation"}, membership) == {
        "aragora.config -> aragora.server"
    }
    assert cic.filter_by_layers(violations, {"infrastructure"}, membership) == {
        "aragora.storage -> aragora.server"
    }


# --- baseline I/O -----------------------------------------------------------


def test_baseline_roundtrip(tmp_path):
    path = tmp_path / "import_contracts_baseline.json"
    violations = {CONTRACT: {"aragora.config -> aragora.server"}}
    cic.write_baseline(path, violations, _REAL_CONFIG)
    loaded = cic.load_baseline(path)
    assert loaded == violations


def test_load_baseline_missing_raises(tmp_path):
    with pytest.raises(cic.CheckerError):
        cic.load_baseline(tmp_path / "does_not_exist.json")


# --- main() behavior (compute monkeypatched) --------------------------------


def _patch_current(monkeypatch, violations: set[str]):
    monkeypatch.setattr(
        cic, "compute_current_violations", lambda config: {CONTRACT: set(violations)}
    )


def test_main_clean_tree_exits_zero(tmp_path, monkeypatch, capsys):
    base = tmp_path / "b.json"
    _write_baseline(base, {"aragora.config -> aragora.knowledge"})
    _patch_current(monkeypatch, {"aragora.config -> aragora.knowledge"})
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(base)])
    assert rc == 0
    assert "no new" in capsys.readouterr().out.lower()


def test_main_new_violation_exits_one_and_names_offender(tmp_path, monkeypatch, capsys):
    base = tmp_path / "b.json"
    _write_baseline(base, {"aragora.storage -> aragora.server"})
    _patch_current(
        monkeypatch,
        {"aragora.storage -> aragora.server", "aragora.config -> aragora.server"},
    )
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(base)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "aragora.config -> aragora.server" in out  # offender named


def test_main_resolved_only_exits_zero(tmp_path, monkeypatch, capsys):
    base = tmp_path / "b.json"
    _write_baseline(base, {"aragora.config -> aragora.server", "aragora.storage -> aragora.server"})
    _patch_current(monkeypatch, {"aragora.storage -> aragora.server"})  # one resolved
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(base)])
    assert rc == 0
    assert "resolved" in capsys.readouterr().out.lower()


def test_main_layers_filter_scopes_failure(tmp_path, monkeypatch):
    base = tmp_path / "b.json"
    _write_baseline(base, set())
    _patch_current(monkeypatch, {"aragora.config -> aragora.server"})  # foundation importer
    # config is foundation, so an infrastructure-scoped check sees no new violation.
    rc_infra = cic.main(
        ["--config", str(_REAL_CONFIG), "--baseline", str(base), "--layers", "infrastructure"]
    )
    assert rc_infra == 0
    rc_foundation = cic.main(
        ["--config", str(_REAL_CONFIG), "--baseline", str(base), "--layers", "foundation"]
    )
    assert rc_foundation == 1


def test_main_missing_baseline_exits_two(tmp_path, monkeypatch):
    _patch_current(monkeypatch, {"aragora.config -> aragora.server"})
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(tmp_path / "nope.json")])
    assert rc == 2


# --- freeze (shrink-only) ---------------------------------------------------


def test_freeze_adopt_writes_baseline(tmp_path, monkeypatch):
    base = tmp_path / "b.json"
    _patch_current(monkeypatch, {"aragora.config -> aragora.server"})
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(base), "--freeze", "--adopt"])
    assert rc == 0
    assert cic.load_baseline(base) == {CONTRACT: {"aragora.config -> aragora.server"}}


def test_freeze_refuses_growth_without_adopt(tmp_path, monkeypatch, capsys):
    base = tmp_path / "b.json"
    _write_baseline(base, {"aragora.storage -> aragora.server"})
    # current adds a brand-new violation not in the baseline
    _patch_current(
        monkeypatch,
        {"aragora.storage -> aragora.server", "aragora.config -> aragora.server"},
    )
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(base), "--freeze"])
    assert rc == 2
    # baseline must be unchanged (shrink-only)
    assert cic.load_baseline(base) == {CONTRACT: {"aragora.storage -> aragora.server"}}


def test_freeze_shrink_succeeds_without_adopt(tmp_path, monkeypatch):
    base = tmp_path / "b.json"
    _write_baseline(base, {"aragora.config -> aragora.server", "aragora.storage -> aragora.server"})
    _patch_current(monkeypatch, {"aragora.storage -> aragora.server"})  # subset => shrink
    rc = cic.main(["--config", str(_REAL_CONFIG), "--baseline", str(base), "--freeze"])
    assert rc == 0
    assert cic.load_baseline(base) == {CONTRACT: {"aragora.storage -> aragora.server"}}
