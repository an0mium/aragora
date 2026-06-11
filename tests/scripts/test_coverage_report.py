"""Tests for scripts/coverage_report.py."""

from __future__ import annotations

import argparse
import sys

import pytest

import scripts.coverage_report as coverage_report


@pytest.mark.parametrize("raw, expected", [("0", 0.0), ("50.5", 50.5), ("100", 100.0)])
def test_coverage_percentage_accepts_valid_percentages(raw: str, expected: float) -> None:
    assert coverage_report.coverage_percentage(raw) == expected


@pytest.mark.parametrize("raw", ["-0.1", "100.1", "nan", "inf", "-inf", "not-a-number"])
def test_coverage_percentage_rejects_impossible_values(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        coverage_report.coverage_percentage(raw)


@pytest.mark.parametrize("raw", ["-1", "101", "nan", "inf", "-inf"])
def test_main_rejects_invalid_min_coverage_before_running(
    raw: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run() -> dict:
        raise AssertionError("coverage should not run for invalid threshold")

    monkeypatch.setattr(coverage_report, "run_coverage", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_report.py", f"--min-coverage={raw}", "--json"],
    )

    with pytest.raises(SystemExit) as exc:
        coverage_report.main()

    assert exc.value.code == 2
    assert "finite percentage from 0 to 100" in capsys.readouterr().err


def test_main_accepts_valid_min_coverage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        coverage_report,
        "run_coverage",
        lambda: {"files": {}, "totals": {"percent_covered": 100.0}},
    )
    monkeypatch.setattr(sys, "argv", ["coverage_report.py", "--min-coverage=100"])

    with pytest.raises(SystemExit) as exc:
        coverage_report.main()

    assert exc.value.code == 0
    assert "OVERALL COVERAGE: 100.0%" in capsys.readouterr().out
