"""Tests for scripts/analyze_test_results.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_test_results.py"
    spec = importlib.util.spec_from_file_location("analyze_test_results", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load analyze_test_results.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_junit_report(path: Path) -> None:
    path.write_text(
        """
<testsuites>
  <testsuite name="pytest" tests="2" failures="1">
    <testcase classname="tests.server.test_app" name="test_ok" time="0.1" />
    <testcase classname="tests.server.test_app" name="test_fail" time="0.2">
      <failure message="boom">Traceback</failure>
    </testcase>
  </testsuite>
</testsuites>
""".strip(),
        encoding="utf-8",
    )


def test_main_returns_nonzero_when_failures_present(tmp_path: Path) -> None:
    module = _load_script_module()
    junit_path = tmp_path / "test-results.xml"
    _write_junit_report(junit_path)

    with patch.object(module.sys, "argv", ["analyze_test_results.py", "--junit", str(junit_path)]):
        assert module.main() == 1


def test_main_exit_zero_flag_makes_analysis_advisory(tmp_path: Path) -> None:
    module = _load_script_module()
    junit_path = tmp_path / "test-results.xml"
    _write_junit_report(junit_path)

    with patch.object(
        module.sys,
        "argv",
        ["analyze_test_results.py", "--junit", str(junit_path), "--exit-zero"],
    ):
        assert module.main() == 0


def test_main_exit_zero_writes_placeholder_when_junit_missing(tmp_path: Path) -> None:
    module = _load_script_module()
    junit_path = tmp_path / "missing.xml"
    output_path = tmp_path / "summary.md"

    with patch.object(
        module.sys,
        "argv",
        [
            "analyze_test_results.py",
            "--junit",
            str(junit_path),
            "--format",
            "markdown",
            "--output",
            str(output_path),
            "--exit-zero",
        ],
    ):
        assert module.main() == 0

    output = output_path.read_text(encoding="utf-8")
    assert "No test results were produced." in output
    assert f"Reason: JUnit report not found: {junit_path}" in output
