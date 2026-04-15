from __future__ import annotations

from scripts import run_proof_first_shift as mod


def test_should_trigger_benchmark_rerun_when_latest_run_is_stale() -> None:
    trigger, reason = mod.should_trigger_benchmark_rerun(
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 123,
            "createdAt": "2026-04-14T00:00:00Z",
            "status": "completed",
            "conclusion": "success",
        },
        has_open_publication_pr=False,
        automation_backlog=0,
        automation_backlog_limit=12,
        last_triggered_run_id=None,
        max_age_hours=1.0,
    )

    assert trigger is True
    assert reason == "stale_publication_window"


def test_should_trigger_benchmark_rerun_respects_backlog_cap() -> None:
    trigger, reason = mod.should_trigger_benchmark_rerun(
        benchmark_mode="hybrid",
        latest_run=None,
        has_open_publication_pr=False,
        automation_backlog=12,
        automation_backlog_limit=12,
        last_triggered_run_id=None,
    )

    assert trigger is False
    assert reason == "automation_backlog_full"


def test_should_restart_service_requires_pending_work_and_budget() -> None:
    assert (
        mod.should_restart_service(
            is_running=False, pending_count=3, restart_count=0, restart_limit=1
        )
        is True
    )
    assert (
        mod.should_restart_service(
            is_running=False, pending_count=3, restart_count=1, restart_limit=1
        )
        is False
    )
    assert mod.should_restart_service(is_running=True, pending_count=3, restart_count=0) is False


def test_classify_benchmark_failure_log_detects_auth_and_publication_failures() -> None:
    assert (
        mod.classify_benchmark_failure_log("codex login required before benchmark publish")
        == "auth_failure"
    )
    assert (
        mod.classify_benchmark_failure_log(
            "resource not accessible by integration during pr creation"
        )
        == "publication_failure"
    )
    assert mod.classify_benchmark_failure_log("unknown shell failure") == "other_failure"
