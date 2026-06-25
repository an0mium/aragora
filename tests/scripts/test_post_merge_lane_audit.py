"""Tests for the shared post-merge lane-audit helper."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from scripts.post_merge_lane_audit import run_post_merge_lane_audit


def _proc(
    args: list[str], payload: dict[str, Any], returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr="",
    )


def _dry_payload(
    *,
    state: str = "MERGED",
    merge_commit: str = "merge-sha",
    finding_count: int = 1,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "finding_count": finding_count,
        "resolved_count": 0,
        "blocked_reason": blocked_reason,
        "owner_steering_text": "",
        "owner_release_commands": [],
        "operator_apply_command": "python3 scripts/resolve_lane_conflicts.py --apply ...",
        "receipt_paths": [],
        "github_state": {
            "available": True,
            "state": state,
            "mergeCommit": merge_commit,
        },
    }


def test_post_merge_lane_audit_dry_run_preserves_resolver_result() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return _proc(args, _dry_payload())

    result = run_post_merge_lane_audit(7435, runner=runner)

    assert len(commands) == 1
    assert "--apply" not in commands[0]
    assert result["audit_ok"] is True
    assert result["audit_applied"] is False
    assert result["operator_apply_command"].startswith("python3 scripts/resolve_lane_conflicts.py")


def test_post_merge_lane_audit_apply_uses_live_merge_commit_guard() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if len(commands) == 1:
            return _proc(args, _dry_payload(merge_commit="live-merge-sha"))
        return _proc(
            args,
            {
                **_dry_payload(merge_commit="live-merge-sha"),
                "resolved_count": 1,
                "receipt_paths": ["/tmp/lane-receipt.json"],
                "blocked_reason": None,
            },
        )

    result = run_post_merge_lane_audit(7435, apply=True, runner=runner)

    assert len(commands) == 2
    assert "--apply" in commands[1]
    assert "--operator-authorized" in commands[1]
    assert commands[1][commands[1].index("--expected-merge-commit") + 1] == "live-merge-sha"
    assert result["audit_ok"] is True
    assert result["audit_applied"] is True
    assert result["receipt_paths"] == ["/tmp/lane-receipt.json"]


def test_post_merge_lane_audit_apply_refuses_unmerged_pr_without_mutation() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return _proc(
            args, _dry_payload(state="OPEN", merge_commit="", blocked_reason="pr_not_merged")
        )

    result = run_post_merge_lane_audit(7435, apply=True, runner=runner)

    assert len(commands) == 1
    assert result["audit_ok"] is False
    assert result["audit_applied"] is False
    assert result["audit_error"] == "pr_not_merged"


def test_post_merge_lane_audit_blocks_payload_with_zero_exit() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return _proc(args, _dry_payload(blocked_reason="invalid_automation_state_root"))

    result = run_post_merge_lane_audit(7435, apply=True, runner=runner)

    assert len(commands) == 1
    assert result["audit_ok"] is False
    assert result["audit_applied"] is False
    assert result["audit_error"] == "invalid_automation_state_root"


def test_post_merge_lane_audit_skips_benign_no_active_lane_result() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return _proc(
            args,
            _dry_payload(
                finding_count=0,
                blocked_reason="no_active_lanes_for_merged_pr",
            ),
        )

    result = run_post_merge_lane_audit(7435, apply=True, runner=runner)

    assert len(commands) == 1
    assert result["audit_ok"] is True
    assert result["audit_applied"] is False
    assert result["audit_apply_skipped_reason"] == "no_active_lanes_for_merged_pr"
