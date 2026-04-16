from __future__ import annotations

from pathlib import Path

import yaml


def _setup_python_safe_fallback_run() -> str:
    action_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "actions"
        / "setup-python-safe"
        / "action.yml"
    )
    action = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    steps = action.get("runs", {}).get("steps", [])
    if not isinstance(steps, list):
        raise AssertionError("setup-python-safe steps not found")
    for step in steps:
        if str(step.get("name", "")) == "Fallback to system Python":
            return str(step.get("run", ""))
    raise AssertionError("Fallback to system Python step not found")


def test_prefers_unpacked_requested_python_before_system_fallback() -> None:
    run = _setup_python_safe_fallback_run()
    unpacked_index = run.index('DISCOVERED_PY="$(find "${RUNNER_TEMP}"')
    search_index = run.index("for cmd in python${{ inputs.python-version }} python3 python; do")
    assert unpacked_index < search_index
    assert "Found unpacked interpreter in RUNNER_TEMP" in run
    assert "Found $cmd:" in run
