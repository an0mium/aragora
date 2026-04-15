from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import run_proof_first_shift as mod


def _run_shift_cycle(
    runtime_state: mod.ProofFirstRuntimeState,
    *,
    process_running_side_effect: list[bool],
    prs: list[dict[str, object]] | None = None,
    latest_run: dict[str, object] | None = None,
    failure_log: str = "",
    benchmark_mode: str = "disabled",
) -> dict[str, object]:
    with (
        patch(
            "scripts.run_proof_first_shift.reconcile_proof_first_queue",
            return_value={"kept": [{"id": 1}], "removed": []},
        ),
        patch(
            "scripts.run_proof_first_shift.collect_boss_lane_snapshot", return_value={"ok": True}
        ),
        patch("scripts.run_proof_first_shift.list_open_prs", return_value=prs or []),
        patch(
            "scripts.run_proof_first_shift.process_running",
            side_effect=process_running_side_effect,
        ),
        patch("scripts.run_proof_first_shift.kickstart_launchd"),
        patch("scripts.run_proof_first_shift.run_merge_arbiter_apply", return_value={"merged": []}),
        patch("scripts.run_proof_first_shift.latest_benchmark_run", return_value=latest_run),
        patch(
            "scripts.run_proof_first_shift.fetch_benchmark_failure_log", return_value=failure_log
        ),
        patch("scripts.run_proof_first_shift.trigger_benchmark_workflow"),
    ):
        return mod.run_shift_cycle(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            benchmark_mode=benchmark_mode,
            automation_backlog_limit=12,
            runtime_state=runtime_state,
        )


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


def test_run_shift_cycle_restores_restart_budget_after_healthy_cycle() -> None:
    state = mod.ProofFirstRuntimeState(boss_restart_count=1, merge_restart_count=1)

    _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        prs=[{"headRefName": "feature/test"}],
    )

    assert state.boss_restart_count == 0
    assert state.merge_restart_count == 0

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[False, False],
        prs=[{"headRefName": "feature/test"}],
    )

    assert state.boss_restart_count == 1
    assert state.merge_restart_count == 1
    assert report["actions"] == ["restart_boss_loop", "restart_merge_arbiter"]


def test_run_shift_cycle_clears_failure_budget_after_successful_benchmark_run() -> None:
    state = mod.ProofFirstRuntimeState(
        auth_failure_count=1,
        publication_failure_count=1,
        last_benchmark_run_id=100,
    )

    _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        latest_run={
            "databaseId": 101,
            "createdAt": "2026-04-15T12:00:00Z",
            "status": "completed",
            "conclusion": "success",
        },
    )

    assert state.auth_failure_count == 0
    assert state.publication_failure_count == 0

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        latest_run={
            "databaseId": 102,
            "createdAt": "2026-04-15T13:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="codex login required before benchmark publish",
    )

    assert state.auth_failure_count == 1
    assert state.publication_failure_count == 0
    assert report["stop_reason"] == ""
