from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.coverage_report as coverage_report


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "coverage_report.py"


def test_load_coverage_json_rejects_malformed_json(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text("{not json}\n", encoding="utf-8")

    try:
        coverage_report.load_coverage_json(coverage_path)
    except coverage_report.CoverageJsonError as exc:
        assert "could not be loaded" in str(exc)
        assert str(coverage_path) in str(exc)
        assert "line 1 column 2" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("malformed coverage JSON should fail closed")


def test_load_coverage_json_rejects_non_object_payload(tmp_path: Path) -> None:
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(["coverage"]) + "\n", encoding="utf-8")

    try:
        coverage_report.load_coverage_json(coverage_path)
    except coverage_report.CoverageJsonError as exc:
        assert "must be an object" in str(exc)
        assert str(coverage_path) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-object coverage JSON should fail closed")


def test_no_run_json_reports_malformed_coverage_without_traceback(tmp_path: Path) -> None:
    (tmp_path / "coverage.json").write_text("{not json}\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--no-run", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "could not be loaded" in proc.stderr
    assert "line 1 column 2" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""
