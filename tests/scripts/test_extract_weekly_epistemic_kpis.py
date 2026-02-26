"""Tests for scripts/extract_weekly_epistemic_kpis.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "extract_weekly_epistemic_kpis.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )


def test_extract_weekly_kpis_passes_with_healthy_fixture(tmp_path: Path) -> None:
    fixture_path = tmp_path / "dashboard.json"
    fixture_path.write_text(
        json.dumps(
            {
                "timestamp": 1700000000,
                "collection_time_ms": 13.7,
                "oracle_stream": {
                    "available": True,
                    "sessions_started": 100,
                    "sessions_completed": 96,
                    "stalls_total": 1,
                    "ttft_avg_ms": 780.4,
                    "ttft_samples": 80,
                },
                "settlement_review": {
                    "available": True,
                    "running": True,
                    "stats": {
                        "success_rate": 0.995,
                        "last_run": "2026-02-26T00:00:00+00:00",
                        "total_runs": 24,
                        "last_result": {"unresolved_due": 0},
                    },
                    "calibration_outcomes": {
                        "correct": 3,
                        "incorrect": 1,
                        "skipped": 0,
                        "deferred": 0,
                        "total": 4,
                        "available": True,
                    },
                },
                "alerts": {"active": [], "total": 0, "available": True},
            }
        ),
        encoding="utf-8",
    )
    json_out = tmp_path / "out.json"
    md_out = tmp_path / "out.md"

    result = _run(
        "--input-json",
        str(fixture_path),
        "--json-out",
        str(json_out),
        "--md-out",
        str(md_out),
        "--strict",
    )

    assert result.returncode == 0
    assert json_out.exists()
    assert md_out.exists()

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["blocking_failures"] == []
    assert payload["kpis"]["settlement_success_rate"]["status"] == "pass"
    assert payload["kpis"]["oracle_stall_rate"]["status"] == "pass"
    assert payload["kpis"]["calibration_updates_realized"]["value"] == 4


def test_extract_weekly_kpis_strict_fails_when_thresholds_breached(tmp_path: Path) -> None:
    fixture_path = tmp_path / "dashboard_bad.json"
    fixture_path.write_text(
        json.dumps(
            {
                "oracle_stream": {
                    "available": True,
                    "sessions_started": 10,
                    "stalls_total": 5,
                    "ttft_avg_ms": 3200,
                    "ttft_samples": 10,
                },
                "settlement_review": {
                    "available": True,
                    "stats": {
                        "success_rate": 0.7,
                        "last_result": {"unresolved_due": 12},
                    },
                    "calibration_outcomes": {
                        "correct": 0,
                        "incorrect": 0,
                        "skipped": 3,
                        "deferred": 2,
                        "total": 5,
                        "available": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = _run("--input-json", str(fixture_path), "--strict")

    assert result.returncode == 1
    summary = json.loads(result.stdout.strip())
    assert summary["passed"] is False
    assert "settlement_success_rate" in summary["blocking_failures"]
    assert "oracle_stall_rate" in summary["blocking_failures"]
    assert "calibration_updates_realized" in summary["blocking_failures"]
