from __future__ import annotations

from pathlib import Path

import yaml


def _auto_pr_publisher_steps() -> list[dict[str, object]]:
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "auto-pr-publisher.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    publish_job = jobs.get("publish-draft-pr", {})
    steps = publish_job.get("steps", [])
    if not isinstance(steps, list):
        raise AssertionError("publish-draft-pr steps not found")
    return [step for step in steps if isinstance(step, dict)]


def _auto_pr_publisher_workflow() -> dict[str, object]:
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "auto-pr-publisher.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        raise AssertionError("auto-pr-publisher workflow not found")
    return workflow


def _workflow_on(workflow: dict[str, object]) -> dict[str, object]:
    on = workflow.get("on", workflow.get(True))
    if not isinstance(on, dict):
        raise AssertionError("auto-pr-publisher triggers not found")
    return on


def test_auto_pr_publisher_triggers_for_benchmark_publication_branches() -> None:
    workflow = _auto_pr_publisher_workflow()
    on = _workflow_on(workflow)
    push = on.get("push")
    assert isinstance(push, dict)
    branches = push.get("branches")
    assert isinstance(branches, list)
    assert "benchmark-truth-publication/**" in branches


def test_auto_pr_publisher_runs_publish_guard_before_pr_creation() -> None:
    steps = _auto_pr_publisher_steps()
    names = [str(step.get("name", "")) for step in steps]

    checkout_index = names.index("Checkout automation branch")
    guard_index = names.index("Preflight automation branch for draft publish")
    publish_index = names.index("Publish draft PR for automation branch")

    assert checkout_index < guard_index < publish_index

    guard_run = str(steps[guard_index].get("run", ""))
    assert "git fetch --no-tags origin main:refs/remotes/origin/main" in guard_run
    assert "bash scripts/automation_pr_preflight.sh origin/main HEAD" in guard_run


def test_auto_pr_publisher_skips_pr_creation_when_publish_guard_blocks() -> None:
    steps = _auto_pr_publisher_steps()
    publish_step = next(
        step for step in steps if step.get("name") == "Publish draft PR for automation branch"
    )

    env = publish_step.get("env")
    assert isinstance(env, dict)
    assert env.get("PUBLISH_GUARD_ALLOW") == "${{ steps.publish_guard.outputs.allow }}"
    assert env.get("PUBLISH_GUARD_REASON") == "${{ steps.publish_guard.outputs.reason }}"

    script = str((publish_step.get("with") or {}).get("script", ""))
    assert "process.env.PUBLISH_GUARD_ALLOW" in script
    assert 'core.setOutput("status", "preflight_failed")' in script


def test_auto_pr_publisher_stops_when_automation_backlog_hits_cap() -> None:
    steps = _auto_pr_publisher_steps()
    publish_step = next(
        step for step in steps if step.get("name") == "Publish draft PR for automation branch"
    )

    script = str((publish_step.get("with") or {}).get("script", ""))
    assert "const backlogLimit = 12;" in script
    assert "const isBenchmarkPublicationHead = isBenchmarkPublicationBranch(branch);" in script
    assert "if (!isBenchmarkPublicationHead && automationBacklog >= backlogLimit)" in script
    assert 'core.setOutput("status", "backlog_full")' in script


def test_auto_pr_publisher_treats_benchmark_publication_branches_as_automation() -> None:
    steps = _auto_pr_publisher_steps()
    publish_step = next(
        step for step in steps if step.get("name") == "Publish draft PR for automation branch"
    )
    script = str((publish_step.get("with") or {}).get("script", ""))
    assert "function isBenchmarkPublicationBranch(ref)" in script
    assert "isBenchmarkPublicationBranch(ref)" in script


def test_auto_pr_publisher_skips_duplicate_benchmark_publication_prs() -> None:
    steps = _auto_pr_publisher_steps()
    publish_step = next(
        step for step in steps if step.get("name") == "Publish draft PR for automation branch"
    )

    script = str((publish_step.get("with") or {}).get("script", ""))
    assert "const benchmarkPublicationOpen = openPulls.some" in script
    assert 'core.setOutput("status", "benchmark_publication_pr_open")' in script
