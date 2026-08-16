"""Focused regression tests for scripts/check_cross_sdk_parity.py baseline resolution.

A named-but-unresolvable baseline (missing file, unreadable/unparseable
content, or ``--strict`` with no baseline at all) must exit with a distinct
configuration-error status naming the attempted path, never silently gate
against an empty baseline. Valid-baseline callers keep identical behavior,
output, and exit codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_cross_sdk_parity.py"

CONFIG_ERROR_EXIT = 2


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=REPO_ROOT,
    )


def _live_baseline(tmp_path: Path) -> Path:
    """Write a baseline grandfathering exactly the current live gaps."""
    report_proc = _run("--json")
    assert report_proc.returncode == 0, report_proc.stderr
    report = json.loads(report_proc.stdout)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "python_only": report["python_only"],
                "typescript_only": report["typescript_only"],
            }
        ),
        encoding="utf-8",
    )
    return baseline


def test_strict_with_missing_baseline_is_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    proc = _run("--strict", "--baseline", str(missing))
    assert proc.returncode == CONFIG_ERROR_EXIT
    assert str(missing) in proc.stderr
    assert "FAILED: Cross-SDK parity regression" not in proc.stdout


def test_missing_baseline_is_config_error_without_strict(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    proc = _run("--baseline", str(missing))
    assert proc.returncode == CONFIG_ERROR_EXIT
    assert str(missing) in proc.stderr


def test_strict_without_baseline_is_config_error() -> None:
    proc = _run("--strict")
    assert proc.returncode == CONFIG_ERROR_EXIT
    assert "--baseline" in proc.stderr
    assert "FAILED: Cross-SDK parity regression" not in proc.stdout


def test_unparseable_baseline_is_config_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    proc = _run("--strict", "--baseline", str(corrupt))
    assert proc.returncode == CONFIG_ERROR_EXIT
    assert str(corrupt) in proc.stderr


def test_non_object_baseline_is_config_error(tmp_path: Path) -> None:
    non_object = tmp_path / "non_object.json"
    non_object.write_text("[]", encoding="utf-8")
    proc = _run("--strict", "--baseline", str(non_object))
    assert proc.returncode == CONFIG_ERROR_EXIT
    assert str(non_object) in proc.stderr


def test_strict_with_valid_baseline_passes_unchanged(tmp_path: Path) -> None:
    baseline = _live_baseline(tmp_path)
    proc = _run("--strict", "--baseline", str(baseline))
    assert proc.returncode == 0
    assert "PASS: No new cross-SDK parity regressions" in proc.stdout
    assert "Baseline regressions: python_only=0 typescript_only=0" in proc.stdout
    assert proc.stderr == ""


def test_plain_invocation_unchanged() -> None:
    proc = _run()
    assert proc.returncode == 0
    assert "Python SDK paths:" in proc.stdout
    assert "Baseline regressions" not in proc.stdout
    assert proc.stderr == ""
