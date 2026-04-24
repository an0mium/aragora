"""Tests for scripts/check_zero_coverage_gate.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_zero_coverage_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_zero_coverage_gate", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_zero_coverage_gate"] = module
    spec.loader.exec_module(module)
    return module


def test_skipped_selector_status_allows_missing_coverage(tmp_path):
    mod = _load_module()
    ok, messages = mod.evaluate_zero_coverage_gate(
        coverage_path=tmp_path / "cov.json",
        baseline_path=tmp_path / "baseline",
        selector_status="skipped",
        changed_python_count=0,
    )
    assert ok is True
    assert messages == [
        "No changed Python files required PR-scoped zero-coverage probing; skipping."
    ]


def test_unmapped_python_changes_fail_even_without_coverage(tmp_path):
    mod = _load_module()
    ok, messages = mod.evaluate_zero_coverage_gate(
        coverage_path=tmp_path / "cov.json",
        baseline_path=tmp_path / "baseline",
        selector_status="unmapped_python_changes",
        changed_python_count=1,
    )
    assert ok is False
    assert "unmapped Python changes" in messages[0]
    assert "must fail the zero-coverage gate" in messages[1]


def test_missing_coverage_fails_for_non_skipped_selector(tmp_path):
    mod = _load_module()
    ok, messages = mod.evaluate_zero_coverage_gate(
        coverage_path=tmp_path / "cov.json",
        baseline_path=tmp_path / "baseline",
        selector_status="dry_run",
        changed_python_count=1,
    )
    assert ok is False
    assert messages == [
        "::error::Coverage data not available at "
        f"{tmp_path / 'cov.json'}; zero-coverage gate cannot verify this change"
    ]


def test_invalid_coverage_json_fails(tmp_path):
    mod = _load_module()
    coverage_path = tmp_path / "cov.json"
    coverage_path.write_text("{not-json")
    ok, messages = mod.evaluate_zero_coverage_gate(
        coverage_path=coverage_path,
        baseline_path=tmp_path / "baseline",
        selector_status="dry_run",
        changed_python_count=1,
    )
    assert ok is False
    assert "invalid JSON" in messages[0]


def test_new_zero_coverage_file_fails(tmp_path):
    mod = _load_module()
    coverage_path = tmp_path / "cov.json"
    coverage_path.write_text(
        json.dumps(
            {
                "files": {
                    "aragora/foo/new_module.py": {
                        "summary": {"percent_covered": 0},
                    }
                }
            }
        )
    )
    ok, messages = mod.evaluate_zero_coverage_gate(
        coverage_path=coverage_path,
        baseline_path=tmp_path / "baseline",
        selector_status="dry_run",
        changed_python_count=1,
    )
    assert ok is False
    assert messages[0] == "::error::New zero-coverage files detected:"
    assert "aragora/foo/new_module.py" in messages[1]


def test_baselined_zero_coverage_file_passes(tmp_path):
    mod = _load_module()
    coverage_path = tmp_path / "cov.json"
    coverage_path.write_text(
        json.dumps(
            {
                "files": {
                    "aragora/foo/legacy_module.py": {
                        "summary": {"percent_covered": 0},
                    }
                }
            }
        )
    )
    baseline_path = tmp_path / "baseline"
    baseline_path.write_text("aragora/foo/legacy_module.py\n")
    ok, messages = mod.evaluate_zero_coverage_gate(
        coverage_path=coverage_path,
        baseline_path=baseline_path,
        selector_status="dry_run",
        changed_python_count=1,
    )
    assert ok is True
    assert messages == ["No new zero-coverage files detected"]


def test_main_cli_returns_failure_for_missing_coverage(monkeypatch, tmp_path, capsys):
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_zero_coverage_gate.py",
            "--coverage-path",
            "cov.json",
            "--changed-python-count",
            "1",
        ],
    )
    exit_code = mod.main()
    assert exit_code == 1
    assert "Coverage data not available" in capsys.readouterr().out
