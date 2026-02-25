"""Tests for scripts/check_epistemic_compliance_regression.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_epistemic_compliance_regression.py"
)


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=dict(os.environ),
    )


def test_checker_passes_on_repository_baseline(tmp_path: Path) -> None:
    json_out = tmp_path / "summary.json"
    md_out = tmp_path / "summary.md"
    result = _run("--strict", "--json-out", str(json_out), "--md-out", str(md_out))

    assert result.returncode == 0
    assert json_out.exists()
    assert md_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "strict_model" in payload["models"]
    assert "adversarial_model" in payload["models"]


def test_checker_fails_when_case_threshold_is_too_strict(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "hype_case",
                        "model": "adversarial_model",
                        "text": "Confidence: 0.99 and this explains everything.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "global": {"min_total_cases": 1},
                "models": {"adversarial_model": {"max_avg_score": 0.2}},
                "cases": {"hype_case": {"max_score": 0.0}},
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        "--strict",
        "--fixtures",
        str(fixtures_path),
        "--baseline",
        str(baseline_path),
    )

    assert result.returncode == 1
    assert "FAIL" in result.stdout
    assert "above threshold" in result.stdout
