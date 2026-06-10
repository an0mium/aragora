"""Tests for the shared post-merge lane-audit helper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts import post_merge_lane_audit as mod


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

    result = mod.run_post_merge_lane_audit(7435, runner=runner)

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

    result = mod.run_post_merge_lane_audit(7435, apply=True, runner=runner)

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

    result = mod.run_post_merge_lane_audit(7435, apply=True, runner=runner)

    assert len(commands) == 1
    assert result["audit_ok"] is False
    assert result["audit_applied"] is False
    assert result["audit_error"] == "pr_not_merged"


def test_post_merge_lane_audit_cli_help_outputs_usage() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "scripts/post_merge_lane_audit.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "usage: post_merge_lane_audit.py" in proc.stdout
    assert "--pr" in proc.stdout
    assert proc.stderr == ""


def test_post_merge_lane_audit_cli_json_uses_requested_options(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(
        pr_number: int,
        *,
        repo_root: Path | None = None,
        apply: bool = False,
        gh_bin: str = "gh",
        runner: Any = None,
    ) -> dict[str, Any]:
        calls.update(
            {
                "pr_number": pr_number,
                "repo_root": repo_root,
                "apply": apply,
                "gh_bin": gh_bin,
                "runner": runner,
            }
        )
        return {
            "audit_ok": True,
            "audit_applied": False,
            "finding_count": 2,
            "resolved_count": 0,
        }

    monkeypatch.setattr(mod, "run_post_merge_lane_audit", fake_run)

    assert (
        mod.main(
            [
                "--pr",
                "7435",
                "--repo-root",
                str(tmp_path),
                "--gh-bin",
                "/usr/bin/false",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["audit_ok"] is True
    assert payload["finding_count"] == 2
    assert calls == {
        "pr_number": 7435,
        "repo_root": tmp_path,
        "apply": False,
        "gh_bin": "/usr/bin/false",
        "runner": None,
    }


def test_post_merge_lane_audit_cli_apply_failure_returns_nonzero(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run(
        pr_number: int,
        *,
        repo_root: Path | None = None,
        apply: bool = False,
        gh_bin: str = "gh",
        runner: Any = None,
    ) -> dict[str, Any]:
        assert pr_number == 7435
        assert apply is True
        return {
            "audit_ok": False,
            "audit_applied": False,
            "audit_error": "merge_commit_missing",
        }

    monkeypatch.setattr(mod, "run_post_merge_lane_audit", fake_run)

    assert mod.main(["--pr", "7435", "--apply"]) == 1
    assert "post-merge lane audit: blocked" in capsys.readouterr().out


def test_post_merge_lane_audit_text_marks_blocked_reason_as_blocked() -> None:
    output = mod._format_text_result(
        {
            "audit_ok": True,
            "audit_applied": False,
            "blocked_reason": "github_state_unavailable",
        },
        pr_number=7435,
    )

    assert output.startswith("post-merge lane audit: blocked\n")
    assert "reason: github_state_unavailable" in output
