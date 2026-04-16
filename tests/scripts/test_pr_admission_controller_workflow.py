from __future__ import annotations

from pathlib import Path

import yaml


def _pr_admission_controller_workflow() -> dict[str, object]:
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "pr-admission-controller.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise AssertionError("pr-admission-controller workflow not found")
    return workflow


def _pr_admission_controller_steps() -> list[dict[str, object]]:
    workflow = _pr_admission_controller_workflow()
    jobs = workflow.get("jobs", {})
    enforce_job = jobs.get("enforce-admission", {})
    steps = enforce_job.get("steps", [])
    if not isinstance(steps, list):
        raise AssertionError("enforce-admission steps not found")
    return [step for step in steps if isinstance(step, dict)]


def test_pr_admission_controller_inlines_checkout_integrity_after_checkout() -> None:
    steps = _pr_admission_controller_steps()
    names = [str(step.get("name", "")) for step in steps]
    checkout_index = names.index("Checkout")
    integrity_index = names.index("Verify checkout integrity")
    assert checkout_index < integrity_index

    integrity_step = steps[integrity_index]
    assert integrity_step.get("uses") is None

    run = str(integrity_step.get("run", ""))
    assert "git sparse-checkout disable || true" in run
    assert 'git fetch --no-tags origin "${GITHUB_SHA:-HEAD}"' in run
    assert 'git archive "${GITHUB_SHA:-HEAD}" | tar -x || git archive HEAD | tar -x' in run
    assert "Repository checkout is incomplete (pyproject.toml missing)" in run
