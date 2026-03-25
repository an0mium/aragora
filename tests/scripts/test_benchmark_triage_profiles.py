"""Tests for scripts/benchmark_triage_profiles.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import benchmark_triage_profiles  # noqa: E402


def test_main_writes_report_and_returns_success(monkeypatch, tmp_path, capsys) -> None:
    report = {
        "fixture_path": "/tmp/fixtures.json",
        "generated_at": "2026-03-25T00:00:00Z",
        "profiles": {
            "baseline": {
                "total_duration_seconds": 12.0,
                "diagnostics_artifact_dir": "/tmp/baseline",
                "meta": {
                    "blocked_count": 1,
                    "suppressed_diagnostics_count": 3,
                    "fast_tier_count": 0,
                    "escalated_count": 0,
                },
            },
            "staged_v1": {
                "total_duration_seconds": 7.0,
                "diagnostics_artifact_dir": "/tmp/staged",
                "meta": {
                    "blocked_count": 1,
                    "suppressed_diagnostics_count": 2,
                    "fast_tier_count": 3,
                    "escalated_count": 1,
                },
            },
        },
        "comparison": {
            "message_count": 4,
            "agreement_rate": 1.0,
            "latency_improvement_pct": 42.0,
            "blocked_rate_delta_pp": 0.0,
            "unsafe_auto_approval_ids": [],
            "acceptance": {
                "decision_agreement": True,
                "latency_improvement": True,
                "blocked_rate_delta": True,
                "unsafe_auto_approval": True,
            },
            "passes_all_thresholds": True,
        },
    }

    async def _fake_run(*args, **kwargs):
        del args, kwargs
        return report

    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(benchmark_triage_profiles, "run_fixture_benchmark", _fake_run)

    exit_code = benchmark_triage_profiles.main(
        ["--fixtures", str(fixture_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    out = capsys.readouterr().out
    assert "Triage Profile Benchmark" in out
    assert "Report written to" in out


def test_main_fails_when_thresholds_missed(monkeypatch, tmp_path) -> None:
    report = {
        "fixture_path": "/tmp/fixtures.json",
        "generated_at": "2026-03-25T00:00:00Z",
        "profiles": {
            "baseline": {
                "total_duration_seconds": 12.0,
                "diagnostics_artifact_dir": "/tmp/baseline",
                "meta": {
                    "blocked_count": 1,
                    "suppressed_diagnostics_count": 3,
                    "fast_tier_count": 0,
                    "escalated_count": 0,
                },
            },
            "staged_v1": {
                "total_duration_seconds": 11.0,
                "diagnostics_artifact_dir": "/tmp/staged",
                "meta": {
                    "blocked_count": 2,
                    "suppressed_diagnostics_count": 4,
                    "fast_tier_count": 1,
                    "escalated_count": 3,
                },
            },
        },
        "comparison": {
            "message_count": 4,
            "agreement_rate": 0.75,
            "latency_improvement_pct": 5.0,
            "blocked_rate_delta_pp": 25.0,
            "unsafe_auto_approval_ids": ["m1"],
            "acceptance": {
                "decision_agreement": False,
                "latency_improvement": False,
                "blocked_rate_delta": False,
                "unsafe_auto_approval": False,
            },
            "passes_all_thresholds": False,
        },
    }

    async def _fake_run(*args, **kwargs):
        del args, kwargs
        return report

    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(benchmark_triage_profiles, "run_fixture_benchmark", _fake_run)

    exit_code = benchmark_triage_profiles.main(
        ["--fixtures", str(fixture_path), "--fail-on-thresholds"]
    )

    assert exit_code == 1
