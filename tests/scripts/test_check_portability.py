"""Tests for scripts/check_portability.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


@pytest.fixture
def cp():
    import check_portability

    return check_portability


def test_find_violations_detects_each_pattern(cp):
    assert cp.find_violations_in_text("home is /Users/alice/dev") == {"users_home"}
    assert cp.find_violations_in_text("exec .venv/bin/python3 foo") == {"venv_python"}
    assert cp.find_violations_in_text("uses: an0mium/aragora@main") == {"legacy_slug"}


def test_find_violations_allows_runner_home(cp):
    # GitHub-hosted macOS runner home is legitimate.
    assert cp.find_violations_in_text("/Users/runner/work/aragora") == set()


def test_find_violations_clean_text(cp):
    assert cp.find_violations_in_text("$HOME/dev synaptent/aragora python3") == set()


def test_scan_paths_detects_skips_and_binary(cp, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs" / "audits").mkdir(parents=True)
    bad = tmp_path / "scripts" / "tool.py"
    bad.write_text('CWD = "/Users/bob/aragora"\n', encoding="utf-8")
    # Always-skip: audit docs legitimately quote the patterns.
    skipped = tmp_path / "docs" / "audits" / "report.md"
    skipped.write_text("/Users/bob and .venv/bin/python\n", encoding="utf-8")
    # Binary file must be ignored without error.
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\x00\xff/Users/bob\x00")

    found = cp.scan_paths(["scripts/tool.py", "docs/audits/report.md", "blob.bin"], root=tmp_path)
    assert found == {"scripts/tool.py": ["users_home"]}


def test_always_skip_covers_pattern_holding_tests(cp):
    # Tests that intentionally hold the patterns as fixtures must be skipped,
    # not flagged (e.g. the installer regression guard from the runtime PR).
    assert cp._is_skipped("tests/scripts/test_launchd_installers.py")
    assert cp._is_skipped("tests/scripts/test_check_portability.py")
    # Regression test asserting the generated runner-health plist is clean holds
    # the `/Users/<name>` literal as negative assertion data, not a real path.
    assert cp._is_skipped("tests/scripts/test_generate_runner_health_plist.py")
    assert cp._is_skipped(".gt/config.json")
    assert not cp._is_skipped("tests/scripts/test_something_else.py")


def test_new_violations_filters_baseline(cp):
    found = {"a.py": ["users_home", "venv_python"], "b.py": ["legacy_slug"]}
    baseline = {"a.py": ["users_home"]}
    assert cp.new_violations(found, baseline) == {
        "a.py": ["venv_python"],
        "b.py": ["legacy_slug"],
    }


def test_baseline_round_trip(cp, tmp_path):
    path = tmp_path / "baseline.json"
    found = {"z.py": ["users_home"], "a.py": ["legacy_slug", "users_home"]}
    cp.write_baseline(found, path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload["files"].keys()) == ["a.py", "z.py"]  # sorted
    assert cp.load_baseline(path) == found


def test_main_passes_when_violation_is_baselined(cp, tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    target = tmp_path / "scripts" / "legacy.py"
    target.write_text('HOME = "/Users/carol/aragora"\n', encoding="utf-8")
    baseline = tmp_path / "scripts" / "portability_baseline.json"
    cp.write_baseline({"scripts/legacy.py": ["users_home"]}, path=baseline)

    monkeypatch.setattr(cp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "BASELINE_PATH", baseline)
    monkeypatch.chdir(tmp_path)

    assert cp.main(["scripts/legacy.py"]) == 0


def test_main_fails_on_new_violation(cp, tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    target = tmp_path / "scripts" / "fresh.py"
    target.write_text('HOME = "/Users/dave/aragora"\n', encoding="utf-8")
    baseline = tmp_path / "scripts" / "portability_baseline.json"
    cp.write_baseline({}, path=baseline)

    monkeypatch.setattr(cp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "BASELINE_PATH", baseline)
    monkeypatch.chdir(tmp_path)

    assert cp.main(["scripts/fresh.py"]) == 1
