"""Tests for ``scripts/pr_check_followup.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


followup = _load_module("pr_check_followup.py")


def _check(
    workflow: str,
    name: str,
    conclusion: str,
    *,
    run_id: str = "123",
    job_id: str = "456",
    started_at: str = "2026-05-23T19:27:02Z",
    completed_at: str = "2026-05-23T19:27:10Z",
) -> dict[str, str]:
    return {
        "workflowName": workflow,
        "name": name,
        "status": "COMPLETED",
        "conclusion": conclusion,
        "detailsUrl": f"https://github.com/synaptent/aragora/actions/runs/{run_id}/job/{job_id}",
        "startedAt": started_at,
        "completedAt": completed_at,
    }


def _pr(
    checks: list[dict[str, str]],
    head: str = "head-sha",
    *,
    mergeable: str = "MERGEABLE",
    merge_state_status: str = "CLEAN",
) -> dict[str, Any]:
    return {
        "number": 7443,
        "headRefOid": head,
        "mergeable": mergeable,
        "mergeStateStatus": merge_state_status,
        "statusCheckRollup": checks,
    }


def _job(
    name: str,
    conclusion: str,
    *,
    job_id: str = "456",
    status: str = "completed",
    steps: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "databaseId": int(job_id),
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "steps": steps or [],
    }


def test_mixed_real_failures_suppress_rerun_commands() -> None:
    result = followup.build_followup_result(
        _pr(
            [
                _check("Tests", "Version Alignment", "FAILURE", run_id="1", job_id="10"),
                _check("Metrics Drift", "check", "CANCELLED", run_id="2", job_id="20"),
            ]
        ),
        allow_rerun_commands=True,
    )

    assert result.action == "repair_failures"
    assert result.rerun_commands == []
    assert "Do not rerun cancelled rows" in result.prompt


def test_only_current_head_early_cancelled_rows_emit_narrow_reruns() -> None:
    result = followup.build_followup_result(
        _pr(
            [
                _check("Metrics Drift", "check", "CANCELLED", run_id="2", job_id="20"),
                _check(
                    "Docs Consistency", "Docs Consistency", "CANCELLED", run_id="3", job_id="30"
                ),
            ]
        ),
        run_data_by_id={
            "2": {"headSha": "head-sha", "jobs": []},
            "3": {"headSha": "head-sha", "jobs": []},
        },
        allow_rerun_commands=True,
    )

    assert result.action == "rerun_cancelled"
    assert result.rerun_commands == [
        "gh run rerun 2 --job 20",
        "gh run rerun 3 --job 30",
    ]
    assert "- gh run rerun 2 --job 20" in result.prompt


def test_in_progress_checks_monitor_without_repair_or_rerun() -> None:
    result = followup.build_followup_result(
        _pr(
            [
                {
                    "workflowName": "Tests",
                    "name": "Type Check",
                    "status": "IN_PROGRESS",
                    "conclusion": "",
                    "detailsUrl": "",
                }
            ]
        )
    )

    assert result.action == "monitor"
    assert result.rerun_commands == []
    assert "monitor #7443" in result.prompt


def test_expected_head_drift_stops_followup() -> None:
    result = followup.build_followup_result(_pr([], head="new-head"), expected_head="old-head")

    assert result.action == "head_drift"
    assert "live head drifted from old-head to new-head" in result.prompt


def test_wait_run_failure_switches_to_failed_job_diagnosis() -> None:
    wait_run = followup.diagnose_wait_run(
        "99",
        {
            "status": "completed",
            "conclusion": "failure",
            "workflowName": "Tests",
            "headSha": "head-sha",
            "jobs": [_job("test-fast", "failure", job_id="101")],
        },
        log_summary_by_job={"101": ["FAILED tests/test_example.py::test_case"]},
    )
    result = followup.build_followup_result(
        _pr([]),
        wait_run=wait_run,
        allow_rerun_commands=True,
    )

    assert result.action == "repair_failures"
    assert result.rerun_commands == []
    assert "gh run view 99 --job 101 --log" in result.prompt
    assert "FAILED tests/test_example.py::test_case" in result.prompt


def test_failed_rerun_after_checkout_is_substantive_blocker() -> None:
    wait_run = followup.diagnose_wait_run(
        "99",
        {
            "status": "completed",
            "conclusion": "failure",
            "workflowName": "Metrics Drift",
            "headSha": "head-sha",
            "jobs": [
                _job(
                    "check",
                    "failure",
                    job_id="101",
                    steps=[
                        {"name": "Set up job", "conclusion": "success"},
                        {"name": "Checkout", "conclusion": "success"},
                        {"name": "Verify drift", "conclusion": "failure"},
                    ],
                )
            ],
        },
        log_summary_by_job={"101": ["Out of date: docs/METRICS.md"]},
    )
    result = followup.build_followup_result(
        _pr([]),
        wait_run=wait_run,
        allow_rerun_commands=True,
    )

    assert result.action == "repair_failures"
    assert result.rerun_commands == []
    assert "failed after checkout during Verify drift" in result.prompt
    assert "Do not rerun cancelled rows" in result.prompt


def test_wait_run_stale_head_does_not_drive_current_head_rerun() -> None:
    wait_run = followup.diagnose_wait_run(
        "99",
        {
            "status": "completed",
            "conclusion": "cancelled",
            "workflowName": "Tests",
            "headSha": "old-head",
            "jobs": [
                _job(
                    "Build Documentation",
                    "cancelled",
                    job_id="101",
                    steps=[
                        {"name": "Set up job", "conclusion": "success"},
                        {"name": "Checkout", "conclusion": "cancelled"},
                    ],
                )
            ],
        },
        pr_head="new-head",
    )
    result = followup.build_followup_result(
        _pr([], head="new-head"),
        wait_run=wait_run,
        allow_rerun_commands=True,
    )

    assert result.action == "stale_wait_run"
    assert result.rerun_commands == []
    assert "waited Actions run belongs to stale head old-head" in result.prompt
    assert "gh run rerun 99 --job 101" not in result.prompt


def test_merge_quorum_model_quorum_failure_emits_evidence_prompt() -> None:
    result = followup.build_followup_result(
        _pr(
            [
                _check(
                    "Aragora Merge Quorum",
                    "aragora-merge-quorum",
                    "FAILURE",
                    run_id="7",
                    job_id="70",
                    started_at="2026-05-27T00:12:07Z",
                    completed_at="2026-05-27T00:13:12Z",
                ),
            ],
            head="evidence-head",
        ),
        run_data_by_id={
            "7": {
                "headSha": "evidence-head",
                "workflowName": "Aragora Merge Quorum",
                "jobs": [
                    _job(
                        "aragora-merge-quorum",
                        "failure",
                        job_id="70",
                        steps=[
                            {"name": "Set up job", "conclusion": "success"},
                            {"name": "Run actions/checkout@v4", "conclusion": "success"},
                            {"name": "Evaluate merge quorum", "conclusion": "failure"},
                        ],
                    )
                ],
            }
        },
        log_summary_by_job={
            "70": [
                "PR #7474 | Tier 2 | status=needs_model_review_quorum | verdict=collect_model_quorum_before_merge",
                "- model quorum incomplete: 0/2 signal(s)",
                "- focused adversarial dogfood evidence is required",
            ]
        },
        allow_rerun_commands=True,
    )

    assert result.action == "collect_model_evidence"
    assert result.rerun_commands == []
    assert (
        "collecting exactly one current-head non-Codex model/dogfood evidence signal"
        in result.prompt
    )
    assert "Post exactly one valid PR comment" in result.prompt
    assert "Do not merge" in result.prompt


def test_summarize_log_keeps_model_quorum_lines() -> None:
    summary = followup.summarize_log(
        "\n".join(
            [
                "noise",
                "PR #7474 | Tier 2 | status=needs_model_review_quorum | verdict=collect_model_quorum_before_merge",
                "  - model quorum incomplete: 0/2 signal(s)",
                "  - focused adversarial dogfood evidence is required",
                "##[error]Merge quorum not met (status=needs_model_review_quorum).",
            ]
        )
    )

    assert any("model quorum incomplete" in line for line in summary)
    assert any("focused adversarial dogfood" in line for line in summary)


def test_wait_run_only_current_head_early_cancellations_emit_reruns() -> None:
    wait_run = followup.diagnose_wait_run(
        "99",
        {
            "status": "completed",
            "conclusion": "cancelled",
            "workflowName": "Tests",
            "headSha": "head-sha",
            "jobs": [
                _job(
                    "Build Documentation",
                    "cancelled",
                    job_id="101",
                    steps=[
                        {"name": "Set up job", "conclusion": "success"},
                        {"name": "Checkout", "conclusion": "cancelled"},
                    ],
                )
            ],
        },
    )
    result = followup.build_followup_result(
        _pr([]),
        wait_run=wait_run,
        allow_rerun_commands=True,
    )

    assert result.action == "rerun_cancelled"
    assert result.rerun_commands == ["gh run rerun 99 --job 101"]
    assert "Build Documentation: early_cancelled" in result.prompt


def test_wait_run_timeout_keeps_prompt_in_monitor_mode() -> None:
    wait_run = followup.diagnose_wait_run(
        "99",
        {
            "status": "in_progress",
            "conclusion": "",
            "workflowName": "Tests",
            "headSha": "head-sha",
            "jobs": [_job("test-fast", "", status="in_progress", job_id="101")],
        },
        timed_out=True,
    )
    result = followup.build_followup_result(_pr([]), wait_run=wait_run)

    assert result.action == "monitor"
    assert "did not settle before the wait timeout" in result.prompt


def test_prompt_always_contains_recursive_convergence_sentences() -> None:
    result = followup.build_followup_result(_pr([]))

    assert followup.INCREMENTAL_PROGRESS_SENTENCE in result.prompt
    assert followup.META_AUTOMATION_SENTENCE in result.prompt


def test_wait_check_waits_for_matching_status_row_then_emits_reruns(monkeypatch: Any) -> None:
    pr_data = _pr(
        [
            {
                "workflowName": "Tests",
                "name": "Baseline Determinism",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": "https://github.com/synaptent/aragora/actions/runs/99/job/101",
                "startedAt": "2026-05-27T02:22:12Z",
                "completedAt": "",
            },
            _check("Metrics Drift", "check", "CANCELLED", run_id="2", job_id="20"),
        ]
    )
    calls: list[list[str]] = []

    def fake_run_gh_json(args: list[str]) -> dict[str, Any]:
        calls.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return pr_data
        if args[:4] == ["gh", "run", "view", "99"]:
            return {
                "status": "completed",
                "conclusion": "success",
                "workflowName": "Tests",
                "headSha": "head-sha",
                "jobs": [_job("Baseline Determinism", "success", job_id="101")],
            }
        if args[:4] == ["gh", "run", "view", "2"]:
            return {
                "headSha": "head-sha",
                "workflowName": "Metrics Drift",
                "jobs": [
                    _job(
                        "check",
                        "cancelled",
                        job_id="20",
                        steps=[
                            {"name": "Set up job", "conclusion": "success"},
                            {"name": "Checkout", "conclusion": "cancelled"},
                        ],
                    )
                ],
            }
        raise AssertionError(f"unexpected gh json call: {args}")

    monkeypatch.setattr(followup, "_run_gh_json", fake_run_gh_json)

    result = followup.fetch_live_result(
        7443,
        expected_head="head-sha",
        include_logs=False,
        allow_rerun_commands=True,
        wait_check="Tests/Baseline Determinism",
        wait_interval_seconds=0,
        wait_timeout_seconds=0,
    )

    assert any(args[:4] == ["gh", "run", "view", "99"] for args in calls)
    assert result.wait_run is not None
    assert result.wait_run.run_id == "99"
    assert result.action == "rerun_cancelled"
    assert result.rerun_commands == ["gh run rerun 2 --job 20"]


def test_green_conflicting_pr_emits_branch_conflict_prompt() -> None:
    result = followup.build_followup_result(
        _pr(
            [_check("Tests", "Baseline Determinism", "SUCCESS")],
            mergeable="CONFLICTING",
            merge_state_status="DIRTY",
        )
    )

    assert result.action == "diagnose_branch_conflict"
    assert "branch conflict" in result.prompt
    assert "Do not merge" in result.prompt


def _required_check(
    name: str,
    bucket: str,
    *,
    workflow: str = "Tests",
    state: str | None = None,
    run_id: str = "123",
    job_id: str = "456",
) -> dict[str, str]:
    return {
        "name": name,
        "bucket": bucket,
        "state": state or bucket.upper(),
        "workflow": workflow,
        "link": f"https://github.com/synaptent/aragora/actions/runs/{run_id}/job/{job_id}",
    }


def _merge_packet(
    *,
    allowed: bool = True,
    not_ready: list[int] | None = None,
    head: str = "head-sha",
) -> dict[str, Any]:
    return {
        "entries": [
            {
                "pr_number": 7443,
                "head_sha": head,
                "admin_squash_allowed": allowed,
                "status": "satisfied" if allowed else "repair_or_wait",
                "verdict": "admin_squash_allowed" if allowed else "not_ready_for_settlement",
            }
        ],
        "admin_squash_order": [7443] if allowed else [],
        "not_ready": not_ready or [],
    }


def test_settlement_guard_required_pending_blocks_packet_authorization() -> None:
    result = followup.build_followup_result(
        _pr([_check("Tests", "Type Check", "SUCCESS")]),
        required_checks=[
            _required_check("aragora-merge-quorum", "pending", workflow="Aragora Merge Quorum"),
        ],
        merge_packet=_merge_packet(allowed=True),
        settlement_guard=True,
    )

    assert result.action == "monitor_required_checks"
    assert result.rerun_commands == []
    assert result.settlement_guard is not None
    assert result.settlement_guard.direct_required_checks_green is False
    assert "required checks are still pending/failing" in result.prompt
    assert "admin squash merge" not in result.prompt


def test_settlement_guard_full_rollup_pending_blocks_stale_quorum_rerun() -> None:
    result = followup.build_followup_result(
        _pr([_check("Aragora Merge Quorum", "aragora-merge-quorum", "CANCELLED")]),
        required_checks=[
            _required_check(
                "aragora-merge-quorum",
                "cancel",
                workflow="Aragora Merge Quorum",
                state="CANCELLED",
                run_id="77",
                job_id="88",
            ),
        ],
        full_rollup_checks=[
            _required_check(
                "aragora-merge-quorum",
                "cancel",
                workflow="Aragora Merge Quorum",
                state="CANCELLED",
                run_id="77",
                job_id="88",
            ),
            _required_check("Baseline Determinism", "pending", workflow="Tests"),
        ],
        merge_packet=_merge_packet(allowed=True),
        settlement_guard=True,
        allow_rerun_commands=True,
    )

    assert result.action == "monitor_full_rollup"
    assert result.rerun_commands == []
    assert "Baseline Determinism" in result.prompt
    assert "gh run rerun" not in result.prompt


def test_settlement_guard_safe_stale_quorum_emits_rerun_only_when_allowed() -> None:
    result = followup.build_followup_result(
        _pr([_check("Aragora Merge Quorum", "aragora-merge-quorum", "CANCELLED")]),
        required_checks=[
            _required_check(
                "aragora-merge-quorum",
                "cancel",
                workflow="Aragora Merge Quorum",
                state="CANCELLED",
                run_id="77",
                job_id="88",
            ),
        ],
        full_rollup_checks=[
            _required_check(
                "aragora-merge-quorum",
                "cancel",
                workflow="Aragora Merge Quorum",
                state="CANCELLED",
                run_id="77",
                job_id="88",
            ),
            _required_check("Type Check", "pass", workflow="Tests"),
        ],
        merge_packet=_merge_packet(allowed=True),
        settlement_guard=True,
        allow_rerun_commands=False,
    )
    allowed = followup.build_followup_result(
        _pr([_check("Aragora Merge Quorum", "aragora-merge-quorum", "CANCELLED")]),
        required_checks=[
            _required_check(
                "aragora-merge-quorum",
                "cancel",
                workflow="Aragora Merge Quorum",
                state="CANCELLED",
                run_id="77",
                job_id="88",
            ),
        ],
        full_rollup_checks=[
            _required_check(
                "aragora-merge-quorum",
                "cancel",
                workflow="Aragora Merge Quorum",
                state="CANCELLED",
                run_id="77",
                job_id="88",
            ),
            _required_check("Type Check", "pass", workflow="Tests"),
        ],
        merge_packet=_merge_packet(allowed=True),
        settlement_guard=True,
        allow_rerun_commands=True,
    )

    assert result.action == "rerun_stale_merge_quorum"
    assert result.rerun_commands == []
    assert allowed.action == "rerun_stale_merge_quorum"
    assert allowed.rerun_commands == ["gh run rerun 77 --job 88"]


def test_settlement_guard_real_non_quorum_failure_repairs_not_reruns() -> None:
    result = followup.build_followup_result(
        _pr([_check("Tests", "Type Check", "FAILURE")]),
        required_checks=[_required_check("Type Check", "fail", workflow="Tests")],
        full_rollup_checks=[_required_check("Type Check", "fail", workflow="Tests")],
        merge_packet=_merge_packet(allowed=True),
        settlement_guard=True,
        allow_rerun_commands=True,
    )

    assert result.action == "repair_failures"
    assert result.rerun_commands == []
    assert "Repair only the real failed checks" in result.prompt


def test_settlement_guard_green_checks_and_packet_authorized_prompts_operator_auth() -> None:
    result = followup.build_followup_result(
        _pr([_check("Tests", "Type Check", "SUCCESS")]),
        required_checks=[_required_check("Type Check", "pass", workflow="Tests")],
        full_rollup_checks=[_required_check("Type Check", "pass", workflow="Tests")],
        merge_packet=_merge_packet(allowed=True),
        settlement_guard=True,
    )

    assert result.action == "ready_for_operator_authorization"
    assert result.settlement_guard is not None
    assert result.settlement_guard.merge_packet_authorizes is True
    assert "I explicitly authorize exact-head admin squash merge for PR #7443" in result.prompt
    assert "gh pr merge 7443 --squash --admin --match-head-commit head-sha" in result.prompt


def test_pr_checks_exit_code_8_still_parses_pending_stdout(monkeypatch: Any) -> None:
    completed = followup.subprocess.CompletedProcess(
        args=["gh", "pr", "checks"],
        returncode=8,
        stdout='[{"name":"Type Check","bucket":"pending","state":"IN_PROGRESS","workflow":"Tests","link":""}]',
        stderr="checks pending",
    )

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return completed

    monkeypatch.setattr(followup.subprocess, "run", fake_run)

    rows = followup._run_json_allow_pending(["gh", "pr", "checks", "7443"])

    assert rows == [
        {
            "name": "Type Check",
            "bucket": "pending",
            "state": "IN_PROGRESS",
            "workflow": "Tests",
            "link": "",
        }
    ]
