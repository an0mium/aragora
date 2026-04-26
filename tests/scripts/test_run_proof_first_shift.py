from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from aragora.nomic.shift_controller import ShiftConfig, ShiftController, ShiftState
from aragora.swarm.shift_ledger import ShiftLedger
from scripts import run_proof_first_shift as mod


def _run_shift_cycle(
    runtime_state: mod.ProofFirstRuntimeState,
    *,
    process_running_side_effect: list[bool],
    prs: list[dict[str, object]] | None = None,
    latest_run: dict[str, object] | None = None,
    failure_log: str = "",
    benchmark_mode: str = "disabled",
    restart_service_side_effect: list[tuple[bool, str]] | None = None,
    trigger_benchmark_side_effect: Exception | None = None,
    ledger: ShiftLedger | None = None,
) -> dict[str, object]:
    boss_restart_patch = (
        patch(
            "scripts.run_proof_first_shift.restart_boss_service",
            side_effect=[
                (ok, detail, "restart_boss_loop") for ok, detail in restart_service_side_effect
            ],
        )
        if restart_service_side_effect is not None
        else patch(
            "scripts.run_proof_first_shift.restart_boss_service",
            return_value=(True, "", "restart_boss_loop"),
        )
    )
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
        boss_restart_patch,
        patch("scripts.run_proof_first_shift.restart_service_via_launchd", return_value=(True, "")),
        patch("scripts.run_proof_first_shift.run_merge_arbiter_apply", return_value={"merged": []}),
        patch("scripts.run_proof_first_shift.latest_benchmark_run", return_value=latest_run),
        patch(
            "scripts.run_proof_first_shift.fetch_benchmark_failure_log", return_value=failure_log
        ),
        patch(
            "scripts.run_proof_first_shift.trigger_benchmark_workflow",
            side_effect=trigger_benchmark_side_effect,
        ),
    ):
        return mod.run_shift_cycle(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            benchmark_mode=benchmark_mode,
            automation_backlog_limit=12,
            runtime_state=runtime_state,
            ledger=ledger,
        )


def _make_shift_controller(repo_root: Path) -> ShiftController:
    checkpoint_dir = repo_root / ".aragora_shifts"
    return ShiftController(
        ShiftConfig(
            repo_path=str(repo_root),
            checkpoint_dir=str(checkpoint_dir),
            require_fresh_assessment=False,
        )
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


def test_should_trigger_benchmark_rerun_ignores_generic_backlog_cap() -> None:
    trigger, reason = mod.should_trigger_benchmark_rerun(
        benchmark_mode="hybrid",
        latest_run=None,
        has_open_publication_pr=False,
        automation_backlog=12,
        automation_backlog_limit=12,
        last_triggered_run_id=None,
    )

    assert trigger is True
    assert reason == "no_prior_run"


def test_should_trigger_benchmark_rerun_waits_for_first_run_visibility() -> None:
    trigger, reason = mod.should_trigger_benchmark_rerun(
        benchmark_mode="hybrid",
        latest_run=None,
        has_open_publication_pr=False,
        automation_backlog=0,
        automation_backlog_limit=12,
        last_triggered_run_id=-1,
    )

    assert trigger is False
    assert reason == "awaiting_first_run_visibility"


def test_should_trigger_benchmark_rerun_waits_for_new_run_after_trigger() -> None:
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
        last_triggered_run_id=123,
        max_age_hours=1.0,
    )

    assert trigger is False
    assert reason == "awaiting_new_benchmark_run"


def test_should_trigger_benchmark_rerun_when_truth_state_drift_detected() -> None:
    fresh_created_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    trigger, reason = mod.should_trigger_benchmark_rerun(
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 123,
            "createdAt": fresh_created_at,
            "status": "completed",
            "conclusion": "success",
        },
        has_open_publication_pr=False,
        automation_backlog=0,
        automation_backlog_limit=12,
        last_triggered_run_id=None,
        truth_state_drift_detected=True,
        max_age_hours=24.0,
    )

    assert trigger is True
    assert reason == "post_generation_issue_state_drift"


def test_should_trigger_benchmark_rerun_skips_when_publication_pr_is_open() -> None:
    trigger, reason = mod.should_trigger_benchmark_rerun(
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 123,
            "createdAt": "2026-04-14T00:00:00Z",
            "status": "completed",
            "conclusion": "success",
        },
        has_open_publication_pr=True,
        automation_backlog=0,
        automation_backlog_limit=12,
        last_triggered_run_id=None,
        max_age_hours=1.0,
    )

    assert trigger is False
    assert reason == "benchmark_pr_open"


def test_has_open_benchmark_publication_pr_matches_branch_namespace() -> None:
    assert mod.has_open_benchmark_publication_pr(
        [
            {"headRefName": None},
            {"headRefName": "codex/benchmark-truth-publication-followup"},
            {"headRefName": "benchmark-truth-publication/24517976139"},
        ]
    )
    assert not mod.has_open_benchmark_publication_pr(
        [
            {"headRefName": "benchmark-truth-publication-fix"},
            {"headRefName": "codex/benchmark-truth-publication/24517976139"},
            {"headRefName": "feature/manual"},
        ]
    )


def test_count_automation_backlog_ignores_draft_prs() -> None:
    prs = [
        {"headRefName": "codex/draft", "isDraft": True},
        {"headRefName": "codex/ready", "isDraft": False},
        {"headRefName": "feature/manual", "isDraft": False},
    ]

    assert mod.count_automation_backlog(prs) == 1
    assert mod.actionable_open_prs(prs) == prs[1:]


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
    assert (
        mod.classify_benchmark_failure_log("HTTP 429: secondary rate limit exceeded")
        == mod.RATE_LIMIT_FAILURE
    )
    assert (
        mod.classify_benchmark_failure_log("permission denied: workflow requires write permission")
        == mod.PERMISSION_MISMATCH_FAILURE
    )
    assert mod.classify_benchmark_failure_log("unknown shell failure") == "other_failure"


def test_kickstart_launchd_returns_timeout_detail_instead_of_raising() -> None:
    with patch(
        "scripts.run_proof_first_shift.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["launchctl", "kickstart", "gui/501/com.aragora.swarm-boss-loop"],
            timeout=30,
            stderr="launchctl timed out",
        ),
    ):
        ok, detail = mod.kickstart_launchd("com.aragora.swarm-boss-loop")

    assert ok is False
    assert detail == "launchctl timed out"


def test_inspect_launchd_service_returns_unavailable_detail_instead_of_raising() -> None:
    with patch(
        "scripts.run_proof_first_shift.subprocess.run",
        side_effect=FileNotFoundError("launchctl not found"),
    ):
        status = mod.inspect_launchd_service("com.aragora.swarm-boss-loop")

    assert status == mod.LaunchdServiceStatus(
        detail="launchctl unavailable for com.aragora.swarm-boss-loop: launchctl not found"
    )


def test_kickstart_launchd_returns_unavailable_detail_instead_of_raising() -> None:
    with patch(
        "scripts.run_proof_first_shift.subprocess.run",
        side_effect=FileNotFoundError("launchctl not found"),
    ):
        ok, detail = mod.kickstart_launchd("com.aragora.swarm-boss-loop")

    assert ok is False
    assert detail == "launchctl unavailable for com.aragora.swarm-boss-loop: launchctl not found"


def test_bind_runtime_state_to_shift_resets_recovery_budgets_for_new_shift() -> None:
    state = mod.ProofFirstRuntimeState(
        recovery_shift_id="old-shift",
        boss_restart_count=1,
        merge_restart_count=1,
        auth_failure_count=1,
        publication_failure_count=1,
        rate_limit_failure_count=1,
        permission_mismatch_count=1,
        runtime_failure_count=1,
        github_outage_count=1,
        recovery_attempt_counts={
            mod.BOSS_RESTART_FAILURE: 1,
            mod.MERGE_RESTART_FAILURE: 1,
            mod.RATE_LIMIT_FAILURE: 1,
        },
        last_benchmark_run_id=123,
        last_triggered_benchmark_run_id=456,
    )

    mod.bind_runtime_state_to_shift(state, "new-shift")

    assert state.recovery_shift_id == "new-shift"
    assert state.boss_restart_count == 0
    assert state.merge_restart_count == 0
    assert state.auth_failure_count == 0
    assert state.publication_failure_count == 0
    assert state.rate_limit_failure_count == 0
    assert state.permission_mismatch_count == 0
    assert state.runtime_failure_count == 0
    assert state.github_outage_count == 0
    assert all(count == 0 for count in state.recovery_attempt_counts.values())
    assert state.last_benchmark_run_id == 123
    assert state.last_triggered_benchmark_run_id == 456


def test_bind_runtime_state_to_shift_preserves_recovery_budgets_for_same_shift() -> None:
    state = mod.ProofFirstRuntimeState(
        recovery_shift_id="same-shift",
        boss_restart_count=1,
        recovery_attempt_counts={mod.BOSS_RESTART_FAILURE: 1},
    )

    mod.bind_runtime_state_to_shift(state, "same-shift")

    assert state.boss_restart_count == 1
    assert state.recovery_attempt_counts[mod.BOSS_RESTART_FAILURE] == 1


def test_save_and_load_runtime_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "runtime_state.json"
    expected_recovery_attempt_counts = dict.fromkeys(mod.RECOVERY_FAILURE_CLASSES, 0)
    expected_recovery_attempt_counts[mod.BOSS_RESTART_FAILURE] = 1
    expected_recovery_attempt_counts[mod.PERMISSION_MISMATCH_FAILURE] = 2
    state = mod.ProofFirstRuntimeState(
        recovery_shift_id="shift-123",
        boss_restart_count=1,
        merge_restart_count=2,
        auth_failure_count=3,
        publication_failure_count=4,
        rate_limit_failure_count=5,
        permission_mismatch_count=6,
        runtime_failure_count=7,
        github_outage_count=8,
        recovery_attempt_counts={
            mod.BOSS_RESTART_FAILURE: 1,
            mod.PERMISSION_MISMATCH_FAILURE: 2,
        },
        last_benchmark_run_id=987,
        last_triggered_benchmark_run_id=654,
    )

    mod.save_runtime_state(path, state)
    loaded = mod.load_runtime_state(path)

    assert loaded == mod.ProofFirstRuntimeState(
        recovery_shift_id="shift-123",
        boss_restart_count=1,
        merge_restart_count=2,
        auth_failure_count=3,
        publication_failure_count=4,
        rate_limit_failure_count=5,
        permission_mismatch_count=6,
        runtime_failure_count=7,
        github_outage_count=8,
        recovery_attempt_counts=expected_recovery_attempt_counts,
        last_benchmark_run_id=987,
        last_triggered_benchmark_run_id=654,
    )


def test_load_runtime_state_returns_fresh_state_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    assert mod.load_runtime_state(missing_path) == mod.ProofFirstRuntimeState()


def test_load_runtime_state_returns_fresh_state_for_invalid_payloads(tmp_path: Path) -> None:
    cases = {
        "empty": "",
        "corrupt": "{not json}",
        "non_dict": "[]",
        "invalid_counts": json.dumps({"boss_restart_count": "oops"}),
    }

    for name, payload in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(payload, encoding="utf-8")
        assert mod.load_runtime_state(path) == mod.ProofFirstRuntimeState()


def test_restore_shift_controller_restores_saved_shift_state(tmp_path: Path) -> None:
    controller = _make_shift_controller(tmp_path)
    checkpoint_dir = Path(controller.config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_state = ShiftState(
        shift_id="shift-test",
        started_at=1.0,
        config={"checkpoint_dir": str(checkpoint_dir)},
        current_cycle=3,
        assessment_id="assessment-123",
    )
    (checkpoint_dir / "latest.json").write_text(
        json.dumps({"shift_state": saved_state.to_dict()}),
        encoding="utf-8",
    )
    controller._state = ShiftState(
        shift_id="stale-shift",
        started_at=2.0,
        config={"checkpoint_dir": str(checkpoint_dir)},
        current_cycle=0,
    )

    restored = mod.restore_shift_controller(controller, checkpoint_dir=checkpoint_dir)

    assert restored == saved_state
    assert controller.state == saved_state


def test_restore_shift_controller_ignores_invalid_checkpoints_without_mutating_state(
    tmp_path: Path,
) -> None:
    cases = {
        "missing": None,
        "empty": "",
        "corrupt": "{not json}",
        "non_dict": "[]",
        "missing_shift_state": json.dumps({"not_shift_state": {}}),
    }

    for name, payload in cases.items():
        repo_root = tmp_path / name
        controller = _make_shift_controller(repo_root)
        checkpoint_dir = Path(controller.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        original_state = ShiftState(
            shift_id=f"original-{name}",
            started_at=2.0,
            config={"checkpoint_dir": str(checkpoint_dir)},
            current_cycle=1,
        )
        controller._state = original_state
        if payload is not None:
            (checkpoint_dir / "latest.json").write_text(payload, encoding="utf-8")

        restored = mod.restore_shift_controller(controller, checkpoint_dir=checkpoint_dir)

        assert restored is None
        assert controller.state == original_state


def test_restart_service_treats_kickstart_timeout_as_success_when_process_appears() -> None:
    with (
        patch(
            "scripts.run_proof_first_shift.kickstart_launchd",
            return_value=(False, "launchctl timed out"),
        ),
        patch("scripts.run_proof_first_shift.process_running", side_effect=[False, True]),
        patch("scripts.run_proof_first_shift.time.sleep"),
    ):
        ok, detail = mod.restart_service_via_launchd(
            label="com.aragora.swarm-boss-loop",
            process_pattern="boss-loop",
            start_timeout_seconds=5,
        )

    assert ok is True
    assert detail == "launchctl timed out"


def test_restart_service_waits_through_launchd_spawn_schedule() -> None:
    status = mod.LaunchdServiceStatus(state="spawn scheduled", minimum_runtime_seconds=300)
    with (
        patch("scripts.run_proof_first_shift.inspect_launchd_service", return_value=status),
        patch(
            "scripts.run_proof_first_shift.kickstart_launchd",
            return_value=(False, "launchctl timed out"),
        ),
        patch("scripts.run_proof_first_shift.wait_for_process", return_value=True) as wait_mock,
    ):
        ok, detail = mod.restart_service_via_launchd(
            label="com.aragora.swarm-boss-loop",
            process_pattern="boss-loop",
        )

    assert ok is True
    assert detail == "launchctl timed out"
    assert wait_mock.call_args.kwargs["timeout_seconds"] >= 360


def test_restart_service_waits_for_successful_kickstart_process_start() -> None:
    with (
        patch("scripts.run_proof_first_shift.kickstart_launchd", return_value=(True, "")),
        patch("scripts.run_proof_first_shift.process_running", side_effect=[False, True]),
        patch("scripts.run_proof_first_shift.time.sleep"),
    ):
        ok, detail = mod.restart_service_via_launchd(
            label="com.aragora.swarm-boss-loop",
            process_pattern="boss-loop",
            start_timeout_seconds=5,
        )

    assert ok is True
    assert detail == ""


def test_restart_service_returns_unavailable_detail_when_launchctl_is_missing() -> None:
    with (
        patch(
            "scripts.run_proof_first_shift.subprocess.run",
            side_effect=FileNotFoundError("launchctl not found"),
        ),
        patch("scripts.run_proof_first_shift.wait_for_process", return_value=False),
        patch("scripts.run_proof_first_shift._read_launchd_failure_detail", return_value=""),
    ):
        ok, detail = mod.restart_service_via_launchd(
            label="com.aragora.swarm-merge-arbiter",
            process_pattern="merge-arbiter",
        )

    assert ok is False
    assert (
        detail == "launchctl unavailable for com.aragora.swarm-merge-arbiter: launchctl not found"
    )


def test_restart_boss_service_uses_direct_bootstrap_when_launchd_service_is_missing() -> None:
    missing = mod.LaunchdServiceStatus(
        detail='Could not find service "com.aragora.swarm-boss-loop" in domain for user gui: 501'
    )
    with (
        patch("scripts.run_proof_first_shift.inspect_launchd_service", return_value=missing),
        patch(
            "scripts.run_proof_first_shift.start_detached_boss_loop",
            return_value=(True, "bootstrapped direct boss loop"),
        ) as bootstrap_mock,
        patch(
            "scripts.run_proof_first_shift.restart_service_via_launchd",
            side_effect=AssertionError(
                "launchd restart should not run when the service is missing"
            ),
        ),
    ):
        ok, detail, action = mod.restart_boss_service(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            process_pattern="boss-loop",
        )

    assert ok is True
    assert detail == "bootstrapped direct boss loop"
    assert action == "bootstrap_boss_loop_direct"
    assert bootstrap_mock.called


def test_restart_boss_service_fails_closed_when_launchctl_is_unavailable() -> None:
    with (
        patch(
            "scripts.run_proof_first_shift.subprocess.run",
            side_effect=FileNotFoundError("launchctl not found"),
        ),
        patch("scripts.run_proof_first_shift.wait_for_process", return_value=False),
        patch("scripts.run_proof_first_shift._read_launchd_failure_detail", return_value=""),
        patch(
            "scripts.run_proof_first_shift.start_detached_boss_loop",
            side_effect=AssertionError(
                "direct bootstrap should not run when launchctl inspection is merely unavailable"
            ),
        ),
    ):
        ok, detail, action = mod.restart_boss_service(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            process_pattern="boss-loop",
        )

    assert ok is False
    assert detail == "launchctl unavailable for com.aragora.swarm-boss-loop: launchctl not found"
    assert action == "restart_boss_loop"


def test_launchd_service_missing_rejects_generic_launchctl_failures() -> None:
    assert (
        mod.launchd_service_missing(
            mod.LaunchdServiceStatus(
                detail="launchctl print failed for com.aragora.swarm-boss-loop"
            )
        )
        is False
    )
    assert (
        mod.launchd_service_missing(
            mod.LaunchdServiceStatus(
                detail="launchctl print timed out for com.aragora.swarm-boss-loop"
            )
        )
        is False
    )
    assert (
        mod.launchd_service_missing(
            mod.LaunchdServiceStatus(
                detail="Operation not permitted while printing com.aragora.swarm-boss-loop"
            )
        )
        is False
    )


def test_restart_boss_service_keeps_generic_launchctl_failures_on_launchd_path() -> None:
    inspection_failure = mod.LaunchdServiceStatus(
        detail="launchctl print failed for com.aragora.swarm-boss-loop"
    )
    with (
        patch(
            "scripts.run_proof_first_shift.inspect_launchd_service",
            return_value=inspection_failure,
        ),
        patch(
            "scripts.run_proof_first_shift.start_detached_boss_loop",
            side_effect=AssertionError(
                "generic launchctl failures should not bootstrap a detached boss loop"
            ),
        ),
        patch(
            "scripts.run_proof_first_shift.restart_service_via_launchd",
            return_value=(False, "launchctl print failed for com.aragora.swarm-boss-loop"),
        ) as restart_mock,
    ):
        ok, detail, action = mod.restart_boss_service(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            process_pattern="boss-loop",
        )

    assert ok is False
    assert detail == "launchctl print failed for com.aragora.swarm-boss-loop"
    assert action == "restart_boss_loop"
    restart_mock.assert_called_once_with(
        label=mod.DEFAULT_BOSS_LABEL,
        process_pattern="boss-loop",
    )


def test_build_direct_boss_loop_command_uses_env_configuration() -> None:
    with patch.dict(
        "os.environ",
        {
            "BOSS_REPO": "synaptent/aragora",
            "TARGET_BRANCH": "release",
            "WORKER_MODEL": "claude",
            "REVIEW_MODEL": "codex",
            "BOSS_LABELS": "boss-ready,autonomous",
            "BOSS_MAX_TICKS": "17",
            "BOSS_INTERVAL_SECONDS": "45",
            "BOSS_MAX_CONSECUTIVE_FAILURES": "9",
            "BOSS_AUTONOMY_MODE": "guided",
            "BOSS_MAX_HOURS": "6",
            "BOSS_MAX_PARALLEL_DISPATCHES": "2",
        },
        clear=False,
    ):
        command = mod.build_direct_boss_loop_command(repo="ignored/repo")

    assert command[:6] == [command[0], "-u", "-m", "aragora.cli.main", "swarm", "boss-loop"]
    assert (
        "--boss-repo" in command
        and command[command.index("--boss-repo") + 1] == "synaptent/aragora"
    )
    assert (
        "--target-branch" in command and command[command.index("--target-branch") + 1] == "release"
    )
    assert command.count("--label") == 2
    assert command[command.index("--max-ticks") + 1] == "17"
    assert command[command.index("--interval") + 1] == "45"
    assert command[command.index("--max-consecutive-failures") + 1] == "9"
    assert command[command.index("--autonomy") + 1] == "guided"
    assert command[command.index("--max-hours") + 1] == "6"
    assert command[command.index("--boss-max-parallel-dispatches") + 1] == "2"


def test_launchd_start_timeout_uses_default_for_non_throttled_state() -> None:
    assert (
        mod.launchd_start_timeout_seconds(mod.LaunchdServiceStatus(state="running"))
        == mod.DEFAULT_LAUNCHD_START_TIMEOUT_SECONDS
    )


def test_inspect_launchd_service_keeps_top_level_state() -> None:
    output = """
gui/501/com.aragora.swarm-boss-loop = {
    state = spawn scheduled
    minimum runtime = 300
    resource coalition = {
        state = active
    }
}
"""
    with patch(
        "scripts.run_proof_first_shift.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["launchctl", "print", "gui/501/com.aragora.swarm-boss-loop"],
            returncode=0,
            stdout=output,
            stderr="",
        ),
    ):
        status = mod.inspect_launchd_service("com.aragora.swarm-boss-loop")

    assert status.state == "spawn scheduled"
    assert status.minimum_runtime_seconds == 300


def test_run_shift_cycle_exhausts_restart_budget_within_shift_window() -> None:
    state = mod.ProofFirstRuntimeState(boss_restart_count=1, merge_restart_count=1)

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        prs=[{"headRefName": "feature/test"}],
    )

    assert report["actions"] == []
    assert state.boss_restart_count == 1
    assert state.merge_restart_count == 1

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[False, False],
        prs=[{"headRefName": "feature/test"}],
    )

    assert report["actions"] == []
    assert report["stop_reason"] == mod.RECOVERY_STOP_REASONS[mod.BOSS_RESTART_FAILURE]
    assert set(report["green_shift_evaluation"]["repeated_failure_classes"]) == {
        mod.BOSS_RESTART_FAILURE,
        mod.MERGE_RESTART_FAILURE,
    }


def test_run_shift_cycle_successful_restart_marks_services_healthy_for_green_shift() -> None:
    state = mod.ProofFirstRuntimeState()

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[False, False],
        prs=[{"headRefName": "feature/test"}],
    )

    assert report["actions"] == ["restart_boss_loop", "restart_merge_arbiter"]
    assert (
        report["green_shift_evaluation"]["criteria"]["services_healthy_or_intentionally_idle"]
        is True
    )
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.BOSS_RESTART_FAILURE][
            "attempts_used"
        ]
        == 1
    )
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.MERGE_RESTART_FAILURE][
            "attempts_used"
        ]
        == 1
    )


def test_run_shift_cycle_clears_failure_budget_after_successful_benchmark_run() -> None:
    state = mod.ProofFirstRuntimeState(
        auth_failure_count=1,
        publication_failure_count=1,
        rate_limit_failure_count=1,
        permission_mismatch_count=1,
        runtime_failure_count=1,
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
    assert state.rate_limit_failure_count == 0
    assert state.permission_mismatch_count == 0
    assert state.runtime_failure_count == 0

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
    assert state.runtime_failure_count == 0
    assert report["stop_reason"] == ""
    assert report["failure_policy"]["auth_failure"]["will_stop"] is False


def test_run_shift_cycle_rate_limit_uses_budget_without_immediate_stop(
    tmp_path: Path,
) -> None:
    state = mod.ProofFirstRuntimeState(last_benchmark_run_id=100)
    ledger = ShiftLedger(path=tmp_path / "test_shift_ledger.jsonl")

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 101,
            "createdAt": "2999-01-01T00:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="HTTP 429: secondary rate limit exceeded",
        ledger=ledger,
    )

    assert state.rate_limit_failure_count == 1
    assert report["actions"] == ["trigger_benchmark:retry_after_failed_run"]
    assert report["stop_reason"] == ""
    assert report["failure_policy"][mod.RATE_LIMIT_FAILURE]["will_stop"] is False
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.RATE_LIMIT_FAILURE][
            "attempts_used"
        ]
        == 1
    )
    assert ledger.get_status_summary()["rate_limit_failures"] == 1


def test_run_shift_cycle_permission_mismatch_uses_budget_without_immediate_stop(
    tmp_path: Path,
) -> None:
    state = mod.ProofFirstRuntimeState(last_benchmark_run_id=100)
    ledger = ShiftLedger(path=tmp_path / "test_shift_ledger.jsonl")

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 101,
            "createdAt": "2999-01-01T00:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="permission denied: workflow requires write permission",
        ledger=ledger,
    )

    assert state.permission_mismatch_count == 1
    assert report["actions"] == ["trigger_benchmark:retry_after_failed_run"]
    assert report["stop_reason"] == ""
    assert report["failure_policy"][mod.PERMISSION_MISMATCH_FAILURE]["will_stop"] is False
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.PERMISSION_MISMATCH_FAILURE][
            "attempts_used"
        ]
        == 1
    )
    assert ledger.get_status_summary()["permission_mismatches"] == 1


def test_run_shift_cycle_reports_restart_failure_instead_of_crashing() -> None:
    state = mod.ProofFirstRuntimeState()

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[False, True],
        restart_service_side_effect=[(False, "launchctl kickstart timed out for boss loop")],
    )

    assert state.boss_restart_count == 1
    assert report["actions"] == ["restart_boss_loop_failed"]
    assert report["stop_reason"] == "BossRestartFailed: launchctl kickstart timed out for boss loop"
    assert report["failure_policy"]["service_failure"]["will_stop"] is True


def test_run_shift_cycle_records_direct_bootstrap_when_launchd_service_is_missing() -> None:
    state = mod.ProofFirstRuntimeState()
    with (
        patch(
            "scripts.run_proof_first_shift.restart_boss_service",
            return_value=(
                True,
                "launchd service missing; bootstrapped direct boss loop pid=123",
                "bootstrap_boss_loop_direct",
            ),
        ),
        patch(
            "scripts.run_proof_first_shift.reconcile_proof_first_queue",
            return_value={"kept": [{"id": 1}], "removed": []},
        ),
        patch(
            "scripts.run_proof_first_shift.collect_boss_lane_snapshot", return_value={"ok": True}
        ),
        patch("scripts.run_proof_first_shift.list_open_prs", return_value=[]),
        patch("scripts.run_proof_first_shift.process_running", side_effect=[False, True]),
        patch("scripts.run_proof_first_shift.restart_service_via_launchd", return_value=(True, "")),
        patch("scripts.run_proof_first_shift.run_merge_arbiter_apply", return_value={"merged": []}),
        patch("scripts.run_proof_first_shift.latest_benchmark_run", return_value=None),
        patch("scripts.run_proof_first_shift.fetch_benchmark_failure_log", return_value=""),
        patch("scripts.run_proof_first_shift.trigger_benchmark_workflow", return_value=None),
    ):
        report = mod.run_shift_cycle(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            benchmark_mode="disabled",
            automation_backlog_limit=12,
            runtime_state=state,
            ledger=None,
        )

    assert report["actions"] == ["bootstrap_boss_loop_direct"]
    assert report["stop_reason"] == ""


def test_run_shift_cycle_stops_cleanly_when_github_is_unavailable(tmp_path: Path) -> None:
    state = mod.ProofFirstRuntimeState()
    ledger = ShiftLedger(path=tmp_path / "test_shift_ledger.jsonl")

    with (
        patch(
            "scripts.run_proof_first_shift.reconcile_proof_first_queue",
            return_value={
                "kept": [],
                "removed": [],
                "github_status": {
                    "available": False,
                    "operation": "list_open_queue_issues",
                    "error": "error connecting to api.github.com",
                },
            },
        ),
        patch(
            "scripts.run_proof_first_shift.collect_boss_lane_snapshot",
            side_effect=AssertionError("snapshot should not run when GitHub is unavailable"),
        ),
        patch(
            "scripts.run_proof_first_shift.list_open_prs",
            side_effect=AssertionError("pr listing should not run when GitHub is unavailable"),
        ),
    ):
        report = mod.run_shift_cycle(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            benchmark_mode="hybrid",
            automation_backlog_limit=12,
            runtime_state=state,
            ledger=ledger,
        )

    assert report["snapshot"] == {}
    assert report["open_pr_count"] == 0
    assert report["automation_backlog"] == 0
    assert report["latest_benchmark_run"] is None
    assert report["actions"] == ["retry_github_outage_next_cycle"]
    assert report["stop_reason"] == ""
    assert state.github_outage_count == 1
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.GITHUB_OUTAGE_FAILURE][
            "attempts_used"
        ]
        == 1
    )


def test_run_shift_cycle_stops_after_repeated_github_outage() -> None:
    state = mod.ProofFirstRuntimeState(
        github_outage_count=1,
        recovery_attempt_counts={mod.GITHUB_OUTAGE_FAILURE: 1},
    )

    with (
        patch(
            "scripts.run_proof_first_shift.reconcile_proof_first_queue",
            return_value={
                "kept": [],
                "removed": [],
                "github_status": {
                    "available": False,
                    "operation": "list_open_queue_issues",
                    "error": "error connecting to api.github.com",
                },
            },
        ),
        patch(
            "scripts.run_proof_first_shift.collect_boss_lane_snapshot",
            side_effect=AssertionError("snapshot should not run when GitHub is unavailable"),
        ),
        patch(
            "scripts.run_proof_first_shift.list_open_prs",
            side_effect=AssertionError("pr listing should not run when GitHub is unavailable"),
        ),
    ):
        report = mod.run_shift_cycle(
            repo_root=Path(".").resolve(),
            repo="synaptent/aragora",
            benchmark_mode="hybrid",
            automation_backlog_limit=12,
            runtime_state=state,
        )

    assert report["actions"] == []
    assert report["stop_reason"] == mod.RECOVERY_STOP_REASONS[mod.GITHUB_OUTAGE_FAILURE]
    assert report["green_shift_evaluation"]["repeated_failure_classes"] == [
        mod.GITHUB_OUTAGE_FAILURE
    ]


def test_run_shift_cycle_fails_closed_after_second_auth_failure() -> None:
    state = mod.ProofFirstRuntimeState(auth_failure_count=1, last_benchmark_run_id=100)

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        latest_run={
            "databaseId": 101,
            "createdAt": "2026-04-15T13:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="codex login required before benchmark publish",
    )

    assert state.auth_failure_count == 2
    assert report["stop_reason"] == "RepeatedAuthFailure: benchmark publication failed auth twice"
    assert report["failure_policy"]["auth_failure"]["will_stop"] is True


def test_run_shift_cycle_fails_closed_after_second_publication_failure() -> None:
    state = mod.ProofFirstRuntimeState(publication_failure_count=1, last_benchmark_run_id=200)

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        latest_run={
            "databaseId": 201,
            "createdAt": "2026-04-15T13:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="resource not accessible by integration during pr creation",
    )

    assert state.publication_failure_count == 2
    assert (
        report["stop_reason"]
        == "RepeatedPublicationFailure: benchmark publication failed PR handoff twice"
    )
    assert report["failure_policy"]["publication_failure"]["will_stop"] is True


def test_run_shift_cycle_fails_closed_after_repeated_rate_limit_failure() -> None:
    state = mod.ProofFirstRuntimeState(
        rate_limit_failure_count=1,
        recovery_attempt_counts={mod.RATE_LIMIT_FAILURE: 1},
        last_benchmark_run_id=300,
    )

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 301,
            "createdAt": "2999-01-01T00:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="API rate limit exceeded",
    )

    assert state.rate_limit_failure_count == 2
    assert report["actions"] == []
    assert report["stop_reason"] == mod.RECOVERY_STOP_REASONS[mod.RATE_LIMIT_FAILURE]
    assert report["failure_policy"][mod.RATE_LIMIT_FAILURE]["will_stop"] is True
    assert report["green_shift_evaluation"]["repeated_failure_classes"] == [mod.RATE_LIMIT_FAILURE]
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.RATE_LIMIT_FAILURE]["exhausted"]
        is True
    )


def test_run_shift_cycle_fails_closed_after_repeated_permission_mismatch() -> None:
    state = mod.ProofFirstRuntimeState(
        permission_mismatch_count=1,
        recovery_attempt_counts={mod.PERMISSION_MISMATCH_FAILURE: 1},
        last_benchmark_run_id=400,
    )

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 401,
            "createdAt": "2999-01-01T00:00:00Z",
            "status": "completed",
            "conclusion": "failure",
        },
        failure_log="permission denied: workflow requires write permission",
    )

    assert state.permission_mismatch_count == 2
    assert report["actions"] == []
    assert report["stop_reason"] == mod.RECOVERY_STOP_REASONS[mod.PERMISSION_MISMATCH_FAILURE]
    assert report["failure_policy"][mod.PERMISSION_MISMATCH_FAILURE]["will_stop"] is True
    assert report["green_shift_evaluation"]["repeated_failure_classes"] == [
        mod.PERMISSION_MISMATCH_FAILURE
    ]
    assert (
        report["green_shift_evaluation"]["recovery_budgets"][mod.PERMISSION_MISMATCH_FAILURE][
            "exhausted"
        ]
        is True
    )


def test_run_shift_cycle_fails_closed_when_benchmark_trigger_runtime_fails() -> None:
    state = mod.ProofFirstRuntimeState()

    report = _run_shift_cycle(
        state,
        process_running_side_effect=[True, True],
        benchmark_mode="hybrid",
        latest_run={
            "databaseId": 123,
            "createdAt": "2026-04-14T00:00:00Z",
            "status": "completed",
            "conclusion": "success",
        },
        trigger_benchmark_side_effect=RuntimeError("gh workflow run failed"),
    )

    assert state.runtime_failure_count == 1
    assert report["actions"] == ["trigger_benchmark_failed:stale_publication_window"]
    assert report["stop_reason"] == "RuntimeFailure: gh workflow run failed"
    assert report["failure_policy"]["runtime_failure"]["will_stop"] is True
