from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "strategic_issue_bridge"


def test_cli_dry_run_heuristic_only() -> None:
    script_path = REPO_ROOT / "scripts" / "generate_strategic_issues.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path.resolve()),
            "--repo",
            str(FIXTURE_ROOT),
            "--dry-run",
            "--heuristic-only",
            "--no-scanner",
            "--max-issues",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Strategic issue candidates" in result.stdout
    assert "RS-01" in result.stdout
    assert "scripts/run_dogfood_benchmark.py" in result.stdout
    assert "scripts/check_benchmark_regression.py" in result.stdout


def test_cli_can_filter_categories_and_emit_json() -> None:
    script_path = REPO_ROOT / "scripts" / "generate_strategic_issues.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path.resolve()),
            "--repo",
            str(FIXTURE_ROOT),
            "--dry-run",
            "--heuristic-only",
            "--no-scanner",
            "--categories",
            "BC",
            "--max-issues",
            "4",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert '"mission_id"' in result.stdout
    assert '"stage_id"' in result.stdout
    assert '"roadmap_refs": [' in result.stdout
    assert '"BC-04"' in result.stdout
    assert '"RS-01"' not in result.stdout
