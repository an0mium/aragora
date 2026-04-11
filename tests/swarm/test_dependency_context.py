"""Tests for dependency context summaries used by supervisor dispatch."""

from __future__ import annotations

from aragora.swarm.dependency_context import (
    build_dependency_context_payload,
    compose_dependency_description,
    dependency_ids_for_work_order,
)


def test_dependency_ids_for_work_order_deduplicates_ids() -> None:
    work_order = {
        "dependency_ids": ["micro-task-1", " micro-task-2 ", "", "micro-task-1", None],
    }

    assert dependency_ids_for_work_order(work_order) == ["micro-task-1", "micro-task-2"]


def test_build_dependency_context_payload_summarizes_completed_dependency() -> None:
    dependency = {
        "work_order_id": "micro-1",
        "pipeline_task_id": "micro-task-1",
        "task_key": "run-1:micro-1",
        "title": "Implementation lane",
        "status": "completed",
        "branch": "codex/swarm-micro-1",
        "head_sha": "abc123",
        "commit_shas": ["abc123", "def456"],
        "changed_paths": ["aragora/swarm/supervisor.py", "tests/swarm/test_supervisor.py"],
        "tests_run": ["python3 -m pytest tests/swarm/test_supervisor.py -q"],
        "verification_results": [
            {
                "command": "python3 -m pytest tests/swarm/test_supervisor.py -q",
                "passed": True,
            }
        ],
    }
    dependent = {
        "dependency_ids": ["micro-task-1"],
    }

    payload = build_dependency_context_payload(dependent, [dependency])

    assert payload["ready_for_dispatch"] is True
    assert payload["base_reference"] == "codex/swarm-micro-1"
    assert payload["base_reference_dependency_id"] == "micro-task-1"
    assert payload["missing_dependency_ids"] == []
    assert payload["terminal_failure"] is None

    context = payload["contexts"][0]
    assert context["dependency_id"] == "micro-task-1"
    assert context["branch"] == "codex/swarm-micro-1"
    assert context["head_sha"] == "abc123"
    assert context["commit_shas"] == ["abc123", "def456"]
    assert context["changed_paths"] == [
        "aragora/swarm/supervisor.py",
        "tests/swarm/test_supervisor.py",
    ]
    assert context["verification_outcomes"] == [
        {
            "command": "python3 -m pytest tests/swarm/test_supervisor.py -q",
            "status": "passed",
            "exit_code": None,
        }
    ]
    assert "do not widen file scope" in payload["prompt_summary"]
    assert "base_ref=codex/swarm-micro-1" in payload["prompt_summary"]
    assert (
        compose_dependency_description("Run validation lane", payload["prompt_summary"])
        == "Run validation lane\n\n" + payload["prompt_summary"]
    )


def test_build_dependency_context_payload_records_terminal_failure_and_missing_dependencies() -> (
    None
):
    dependency = {
        "work_order_id": "micro-1",
        "pipeline_task_id": "micro-task-1",
        "title": "Implementation lane",
        "status": "discarded",
        "failure_reason": "work_order_leasing_failed",
        "dispatch_error": "worktree creation failed",
        "branch": "codex/swarm-micro-1",
        "changed_paths": ["aragora/swarm/supervisor.py"],
    }
    dependent = {
        "dependency_ids": ["micro-task-1", "micro-task-2"],
    }

    payload = build_dependency_context_payload(dependent, [dependency])

    assert payload["ready_for_dispatch"] is False
    assert payload["base_reference"] is None
    assert payload["missing_dependency_ids"] == ["micro-task-2"]
    assert payload["terminal_failure"] == {
        "dependency_id": "micro-task-1",
        "dependency_status": "discarded",
        "dependency_reason": "work_order_leasing_failed",
    }

    context = payload["contexts"][0]
    assert context["status"] == "discarded"
    assert context["failure_reason"] == "work_order_leasing_failed"
    assert "missing dependencies: micro-task-2" in payload["prompt_summary"]
    assert "blocked_reason: worktree creation failed" in payload["prompt_summary"]


def test_build_dependency_context_payload_does_not_mark_recoverable_needs_human_as_terminal() -> (
    None
):
    dependency = {
        "work_order_id": "micro-1",
        "pipeline_task_id": "micro-task-1",
        "title": "Implementation lane",
        "status": "needs_human",
        "failure_reason": "stale_lease_reaped",
        "branch": "codex/swarm-micro-1",
    }

    payload = build_dependency_context_payload({"dependency_ids": ["micro-task-1"]}, [dependency])

    assert payload["ready_for_dispatch"] is False
    assert payload["terminal_failure"] is None
    assert payload["contexts"][0]["status"] == "needs_human"


def test_build_dependency_context_payload_marks_archived_needs_human_as_terminal() -> None:
    dependency = {
        "work_order_id": "micro-1",
        "pipeline_task_id": "micro-task-1",
        "title": "Implementation lane",
        "status": "needs_human",
        "failure_reason": "stale_lease_reaped",
        "branch": "codex/swarm-micro-1",
        "metadata": {
            "archived_due_to": "stale_lease_reaped",
        },
    }

    payload = build_dependency_context_payload({"dependency_ids": ["micro-task-1"]}, [dependency])

    assert payload["ready_for_dispatch"] is False
    assert payload["terminal_failure"] == {
        "dependency_id": "micro-task-1",
        "dependency_status": "needs_human",
        "dependency_reason": "stale_lease_reaped",
    }
