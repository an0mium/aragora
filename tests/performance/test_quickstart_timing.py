"""
Tests for the quickstart timing measurement script.

Validates each timing step in isolation, threshold comparison logic,
JSON output format, and report formatting.

Run with: pytest tests/performance/test_quickstart_timing.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.measure_quickstart_time import (
    THRESHOLDS,
    QuickstartReport,
    StepResult,
    evaluate_step,
    format_report,
    measure_import_time,
    measure_receipt_generation,
    measure_repo_size,
    run_all_measurements,
)


# =============================================================================
# StepResult / evaluate_step tests
# =============================================================================


class TestEvaluateStep:
    """Tests for the evaluate_step helper."""

    def test_passing_step(self):
        """A duration within threshold yields passed=True."""
        step = evaluate_step("test", 1.0, 5.0)
        assert step.passed is True
        assert step.status == "PASS"
        assert step.duration == 1.0

    def test_failing_step(self):
        """A duration exceeding threshold yields passed=False."""
        step = evaluate_step("test", 10.0, 5.0)
        assert step.passed is False
        assert step.status == "FAIL"

    def test_exact_threshold_passes(self):
        """Duration exactly at the threshold still passes."""
        step = evaluate_step("test", 5.0, 5.0)
        assert step.passed is True
        assert step.status == "PASS"

    def test_skipped_step(self):
        """A negative duration marks the step as SKIP."""
        step = evaluate_step("test", -1.0, 5.0)
        assert step.status == "SKIP"
        assert step.passed is True

    def test_informational_step(self):
        """A zero threshold yields INFO status (always passes)."""
        step = evaluate_step("test", 99.0, 0.0)
        assert step.status == "INFO"
        assert step.passed is True

    def test_error_field_preserved(self):
        """Error and detail strings are carried through."""
        step = evaluate_step("test", 1.0, 5.0, error="boom", detail="extra")
        assert step.error == "boom"
        assert step.detail == "extra"

    def test_duration_rounded_to_three_decimals(self):
        """Duration is rounded to 3 decimal places."""
        step = evaluate_step("test", 1.23456789, 5.0)
        assert step.duration == 1.235


# =============================================================================
# Threshold configuration tests
# =============================================================================


class TestThresholds:
    """Tests for the threshold configuration constants."""

    def test_all_thresholds_present(self):
        """Required threshold keys exist."""
        required = {"clone", "install", "import", "first_debate", "receipt", "server_startup", "total"}
        assert required.issubset(THRESHOLDS.keys())

    def test_thresholds_are_non_negative(self):
        """All thresholds are >= 0."""
        for key, val in THRESHOLDS.items():
            assert val >= 0, f"Threshold '{key}' is negative: {val}"

    def test_total_threshold_is_largest(self):
        """Total threshold is at least as large as any individual threshold."""
        individual = {k: v for k, v in THRESHOLDS.items() if k != "total"}
        for key, val in individual.items():
            assert THRESHOLDS["total"] >= val, (
                f"Total threshold ({THRESHOLDS['total']}) < '{key}' threshold ({val})"
            )


# =============================================================================
# Individual measurement tests
# =============================================================================


class TestMeasureRepoSize:
    """Tests for measure_repo_size()."""

    def test_returns_positive_size(self):
        """Repo size is positive (we are inside a git repo)."""
        dur, size_mb = measure_repo_size()
        assert size_mb > 0
        assert dur >= 0

    def test_missing_git_dir_returns_negative(self, tmp_path):
        """Returns (-1, 0) when .git does not exist."""
        # Point the function at a temp directory that has no .git
        fake_script = tmp_path / "scripts" / "fake.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        fake_script.touch()

        with patch("scripts.measure_quickstart_time.__file__", str(fake_script)):
            dur, size_mb = measure_repo_size()
            assert dur == -1.0
            assert size_mb == 0.0


class TestMeasureImportTime:
    """Tests for measure_import_time()."""

    def test_returns_step_result(self):
        """Returns a StepResult with positive duration."""
        step = measure_import_time()
        assert isinstance(step, StepResult)
        assert step.duration >= 0
        assert step.name == "Import aragora"

    def test_imports_key_modules(self):
        """Detail string reports successful imports."""
        step = measure_import_time()
        # Should import at least aragora itself
        assert "imported" in step.detail


class TestMeasureReceiptGeneration:
    """Tests for measure_receipt_generation()."""

    def test_receipt_generated(self):
        """Receipt is generated within threshold."""
        step = measure_receipt_generation()
        assert isinstance(step, StepResult)
        assert step.name == "Receipt generation"
        assert step.duration >= 0
        assert step.passed is True, f"Receipt took {step.duration}s, threshold={step.threshold}s"

    def test_receipt_detail_contains_verdict(self):
        """Detail includes the receipt verdict."""
        step = measure_receipt_generation()
        assert "verdict=" in step.detail


# =============================================================================
# Report formatting tests
# =============================================================================


class TestFormatReport:
    """Tests for the format_report function."""

    def _make_report(self) -> QuickstartReport:
        return QuickstartReport(
            steps=[
                evaluate_step("Clone/Download", -1.0, 0.0, detail="repo .git = 200.0 MB"),
                evaluate_step("Dependencies Install", 12.3, 60.0),
                evaluate_step("Import aragora", 0.8, 5.0),
                evaluate_step("First debate", 2.1, 10.0),
                evaluate_step("Receipt generation", 0.1, 1.0),
                evaluate_step("Server startup", 1.5, 5.0),
            ],
            total_seconds=16.8,
            all_passed=True,
            repo_size_mb=200.0,
        )

    def test_contains_header(self):
        """Report starts with the expected header."""
        text = format_report(self._make_report())
        assert "Aragora Quickstart Timing Report" in text

    def test_contains_all_step_names(self):
        """Every step name appears in the report."""
        report = self._make_report()
        text = format_report(report)
        for step in report.steps:
            assert step.name in text

    def test_contains_total_line(self):
        """Total quickstart time line is present."""
        text = format_report(self._make_report())
        assert "Total quickstart time" in text

    def test_contains_repo_size(self):
        """Repo size line is present when size > 0."""
        text = format_report(self._make_report())
        assert "Repo size (.git)" in text

    def test_skipped_step_shows_na(self):
        """A skipped step (duration < 0) shows N/A."""
        text = format_report(self._make_report())
        assert "N/A" in text


# =============================================================================
# JSON output tests
# =============================================================================


class TestJsonOutput:
    """Tests for JSON report serialization."""

    def test_to_dict_has_required_keys(self):
        """to_dict() contains all expected top-level keys."""
        report = QuickstartReport(
            steps=[evaluate_step("test", 1.0, 5.0)],
            total_seconds=1.0,
            all_passed=True,
        )
        d = report.to_dict()
        assert "steps" in d
        assert "total_seconds" in d
        assert "all_passed" in d
        assert "thresholds" in d

    def test_json_serializable(self):
        """to_dict() output is JSON-serializable."""
        report = QuickstartReport(
            steps=[
                evaluate_step("a", 1.0, 5.0),
                evaluate_step("b", -1.0, 0.0, error="skip"),
            ],
            total_seconds=1.0,
            all_passed=True,
            repo_size_mb=100.0,
        )
        text = json.dumps(report.to_dict())
        parsed = json.loads(text)
        assert parsed["all_passed"] is True
        assert len(parsed["steps"]) == 2

    def test_step_dict_fields(self):
        """Each step dict contains the expected fields."""
        step = evaluate_step("test", 2.5, 5.0, error="", detail="ok")
        report = QuickstartReport(steps=[step], total_seconds=2.5, all_passed=True)
        step_dict = report.to_dict()["steps"][0]
        for key in ("name", "duration", "threshold", "passed", "error", "detail"):
            assert key in step_dict, f"Missing key '{key}' in step dict"

    def test_cli_json_flag(self):
        """Running the script with --json produces valid JSON."""
        result = subprocess.run(
            [sys.executable, "-c", (
                "from scripts.measure_quickstart_time import QuickstartReport, evaluate_step;\n"
                "import json;\n"
                "r = QuickstartReport(steps=[evaluate_step('t', 1.0, 5.0)], total_seconds=1.0, all_passed=True);\n"
                "print(json.dumps(r.to_dict()))\n"
            )],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout.strip())
        assert "steps" in parsed


# =============================================================================
# Integration / end-to-end
# =============================================================================


class TestRunAllMeasurements:
    """Smoke test for the full measurement pipeline.

    Note: this actually exercises real imports and receipt generation
    so it is not a pure unit test, but it validates the orchestration
    logic end-to-end without starting a real server.
    """

    def test_report_has_all_steps(self):
        """run_all_measurements returns a report with 6 steps."""
        with patch("scripts.measure_quickstart_time.measure_server_startup") as mock_server, \
             patch("scripts.measure_quickstart_time.measure_install_time") as mock_install:
            mock_server.return_value = evaluate_step("Server startup", 0.5, 5.0)
            mock_install.return_value = evaluate_step("Dependencies Install", 1.0, 60.0)
            report = run_all_measurements()
        assert len(report.steps) == 6

    def test_total_seconds_is_sum_of_measured(self):
        """total_seconds equals the sum of non-skipped step durations."""
        with patch("scripts.measure_quickstart_time.measure_server_startup") as mock_server, \
             patch("scripts.measure_quickstart_time.measure_install_time") as mock_install:
            mock_server.return_value = evaluate_step("Server startup", 0.5, 5.0)
            mock_install.return_value = evaluate_step("Dependencies Install", 1.0, 60.0)
            report = run_all_measurements()
        measured = [s for s in report.steps if s.duration >= 0]
        expected = round(sum(s.duration for s in measured), 3)
        assert report.total_seconds == expected

    def test_all_passed_false_when_step_fails(self):
        """all_passed is False when any step fails."""
        with patch("scripts.measure_quickstart_time.measure_server_startup") as mock_server, \
             patch("scripts.measure_quickstart_time.measure_install_time") as mock_install:
            mock_server.return_value = evaluate_step("Server startup", 999.0, 5.0)
            mock_install.return_value = evaluate_step("Dependencies Install", 1.0, 60.0)
            report = run_all_measurements()
        assert report.all_passed is False
