from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/openapi.yml"
STEP_NAME = "Validate handler route coverage"


def _route_coverage_run_block() -> str:
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(workflow, dict)

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict) and step.get("name") == STEP_NAME:
                return str(step["run"])

    raise AssertionError(f"workflow step not found: {STEP_NAME}")


def test_route_coverage_pipeline_fails_closed_and_preserves_report(tmp_path: Path) -> None:
    run_block = _route_coverage_run_block()
    commands = [line.strip() for line in run_block.splitlines() if line.strip()]

    assert commands[0] == "set -o pipefail"
    validator_index = next(
        index for index, command in enumerate(commands) if "validate_openapi_routes.py" in command
    )
    assert validator_index > 0
    assert "--json | tee /tmp/route-coverage.json" in run_block

    report = tmp_path / "route-coverage.json"
    payload = '{"coverage_percentage": 97.0}\n'
    producer = f"(printf %s {shlex.quote(payload)}; exit 7)"
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            f"{commands[0]}\n{producer} | tee {shlex.quote(str(report))} >/dev/null",
        ],
        check=False,
    )

    assert result.returncode == 7
    assert report.read_text(encoding="utf-8") == payload
