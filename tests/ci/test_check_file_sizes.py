"""Unit tests for scripts/ci/check_file_sizes.py.

These exercise the pure measurement/offender/freeze logic and the ``main`` CLI
wiring against a temporary fake checkout (``REPO_ROOT`` and
``list_tracked_py_files`` are monkeypatched) so the tests stay fast and never
depend on the real census. The end-to-end behavior against the real ``aragora``
package (green on clean tree, oversized-newcomer tamper, baseline shrink-only)
is covered by the VAL-P0-004/005 acceptance checks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_PATH = REPO_ROOT / "scripts" / "ci" / "check_file_sizes.py"

_spec = importlib.util.spec_from_file_location("check_file_sizes", _CHECKER_PATH)
cfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfs)


def _make_file(root: Path, rel: str, lines: int) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# pad\n" * lines, encoding="utf-8")


# --- pure logic -------------------------------------------------------------


def test_count_lines_matches_splitlines(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    assert cfs.count_lines(f) == 3
    # missing file counts as 0, never raises
    assert cfs.count_lines(tmp_path / "missing.py") == 0


def test_find_offenders_flags_new_and_skips_baselined():
    oversized = {"aragora/new_big.py": 2500, "aragora/known_big.py": 2400}
    baseline = {"aragora/known_big.py"}
    offenders = cfs.find_offenders(oversized, baseline)
    assert offenders == {"aragora/new_big.py": 2500}


def test_measure_oversized_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    _make_file(tmp_path, "aragora/big.py", 2100)
    _make_file(tmp_path, "aragora/exact.py", cfs.LIMIT)  # exactly at limit: not over
    _make_file(tmp_path, "aragora/small.py", 10)
    oversized = cfs.measure_oversized(["aragora/big.py", "aragora/exact.py", "aragora/small.py"])
    assert oversized == {"aragora/big.py": 2100}


# --- CLI: check mode --------------------------------------------------------


def test_main_green_when_oversized_baselined(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cfs, "list_tracked_py_files", lambda: ["aragora/big.py"])
    _make_file(tmp_path, "aragora/big.py", 2100)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"files": {"aragora/big.py": 2100}}), encoding="utf-8")
    assert cfs.main(["--baseline", str(baseline)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_main_fails_and_names_unbaselined_oversized(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cfs, "list_tracked_py_files", lambda: ["aragora/newcomer.py"])
    _make_file(tmp_path, "aragora/newcomer.py", 2100)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"files": {}}), encoding="utf-8")
    assert cfs.main(["--baseline", str(baseline)]) == 1
    assert "aragora/newcomer.py" in capsys.readouterr().out


def test_main_missing_baseline_is_usage_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cfs, "list_tracked_py_files", lambda: [])
    assert cfs.main(["--baseline", str(tmp_path / "nope.json")]) == 2


# --- CLI: freeze (shrink-only) ---------------------------------------------


def test_freeze_refuses_to_grow_without_adopt(tmp_path, monkeypatch):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cfs, "list_tracked_py_files", lambda: ["aragora/big.py"])
    _make_file(tmp_path, "aragora/big.py", 2100)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"files": {}}), encoding="utf-8")  # empty -> big.py is new
    # exit code 2 (shrink-only violation), baseline unchanged
    assert cfs.main(["--freeze", "--baseline", str(baseline)]) == 2
    assert json.loads(baseline.read_text())["files"] == {}


def test_freeze_adopt_writes_full_census(tmp_path, monkeypatch):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        cfs, "list_tracked_py_files", lambda: ["aragora/big.py", "aragora/small.py"]
    )
    _make_file(tmp_path, "aragora/big.py", 2100)
    _make_file(tmp_path, "aragora/small.py", 10)
    baseline = tmp_path / "baseline.json"
    assert cfs.main(["--freeze", "--adopt", "--baseline", str(baseline)]) == 0
    written = json.loads(baseline.read_text())
    assert written["files"] == {"aragora/big.py": 2100}
    assert written["limit"] == cfs.LIMIT


def test_freeze_allows_shrink(tmp_path, monkeypatch):
    monkeypatch.setattr(cfs, "REPO_ROOT", tmp_path)
    # baseline has two entries; only one is still oversized -> shrink is allowed
    monkeypatch.setattr(
        cfs, "list_tracked_py_files", lambda: ["aragora/big.py", "aragora/shrunk.py"]
    )
    _make_file(tmp_path, "aragora/big.py", 2100)
    _make_file(tmp_path, "aragora/shrunk.py", 50)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"files": {"aragora/big.py": 2100, "aragora/shrunk.py": 2400}}),
        encoding="utf-8",
    )
    assert cfs.main(["--freeze", "--baseline", str(baseline)]) == 0
    assert json.loads(baseline.read_text())["files"] == {"aragora/big.py": 2100}
