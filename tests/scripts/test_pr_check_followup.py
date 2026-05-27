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


def _pr(checks: list[dict[str, str]], head: str = "head-sha") -> dict[str, Any]:
    return {"number": 7443, "headRefOid": head, "statusCheckRollup": checks}


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


def test_prompt_always_contains_incremental_progress_sentence() -> None:
    result = followup.build_followup_result(_pr([]))

    assert followup.INCREMENTAL_PROGRESS_SENTENCE in result.prompt


def test_rest_source_fetches_checks_without_graphql_pr_view(monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    def fake_run_gh_json(args: list[str]) -> dict[str, Any] | list[dict[str, Any]]:
        calls.append(args)
        if args == ["gh", "api", "rate_limit"]:
            return {
                "resources": {
                    "core": {
                        "limit": 5000,
                        "remaining": 4990,
                        "reset": 1779853680,
                    }
                }
            }
        if args == ["gh", "api", "repos/synaptent/aragora/pulls/7447"]:
            return {
                "number": 7447,
                "state": "open",
                "draft": False,
                "html_url": "https://github.com/synaptent/aragora/pull/7447",
                "head": {
                    "ref": "codex/review-queue-evidence-lint-20260523",
                    "sha": "head-sha",
                },
            }
        if args[-1] == "repos/synaptent/aragora/commits/head-sha/check-runs?per_page=100":
            return {
                "check_runs": [
                    {
                        "name": "aragora-merge-quorum",
                        "status": "completed",
                        "conclusion": "success",
                        "details_url": (
                            "https://github.com/synaptent/aragora/actions/runs/123/job/456"
                        ),
                        "started_at": "2026-05-27T02:47:34Z",
                        "completed_at": "2026-05-27T02:48:30Z",
                        "check_suite": {"head_sha": "head-sha"},
                    }
                ]
            }
        if args[-1] == "repos/synaptent/aragora/commits/head-sha/statuses?per_page=100":
            return [
                {
                    "context": "aragora/human-settlement",
                    "state": "success",
                    "target_url": "https://github.com/synaptent/aragora/pull/7447",
                    "updated_at": "2026-05-27T02:35:17Z",
                }
            ]
        raise AssertionError(f"unexpected gh JSON call: {args}")

    monkeypatch.setattr(followup, "_repo_slug_from_git", lambda: "synaptent/aragora")
    monkeypatch.setattr(followup, "_run_gh_json", fake_run_gh_json)

    result = followup.fetch_live_result_rest(
        7447,
        expected_head="head-sha",
        include_logs=False,
        allow_rerun_commands=False,
        min_rest_remaining=5,
    )

    assert result.source == "rest"
    assert result.action == "green"
    assert result.rate_limit is not None
    assert result.rate_limit["remaining"] == 4990
    assert not any(call[:3] == ["gh", "pr", "view"] for call in calls)


def test_rest_source_stops_before_exhausting_core_rate_limit(monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    def fake_run_gh_json(args: list[str]) -> dict[str, Any]:
        calls.append(args)
        assert args == ["gh", "api", "rate_limit"]
        return {
            "resources": {
                "core": {
                    "limit": 5000,
                    "remaining": 3,
                    "reset": 1779853680,
                }
            }
        }

    monkeypatch.setattr(followup, "_repo_slug_from_git", lambda: "synaptent/aragora")
    monkeypatch.setattr(followup, "_run_gh_json", fake_run_gh_json)

    result = followup.fetch_live_result_rest(
        7447,
        expected_head="head-sha",
        include_logs=False,
        allow_rerun_commands=False,
        min_rest_remaining=5,
    )

    assert result.action == "github_rate_limited"
    assert result.source == "rest"
    assert result.rate_limit is not None
    assert result.rate_limit["remaining"] == 3
    assert calls == [["gh", "api", "rate_limit"]]
