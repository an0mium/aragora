"""Shared post-merge lane-audit helper.

This wraps ``scripts/resolve_lane_conflicts.py --merged-pr-lane-audit`` so
settlement paths can report stale active lane rows after a PR lands, while
keeping the default behavior dry-run and non-destructive.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]
DRY_RUN_ZERO_EXIT_BLOCKERS = {"invalid_automation_state_root"}


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    runner: Runner | None,
) -> subprocess.CompletedProcess[str]:
    if runner is not None:
        return runner(args)
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _parse_json(stdout: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _command_error(
    *,
    command: list[str],
    proc: subprocess.CompletedProcess[str] | None,
    message: str,
    apply_requested: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "audit_ok": False,
        "audit_applied": False,
        "audit_apply_requested": apply_requested,
        "audit_command": command,
        "audit_error": message,
    }
    if proc is not None:
        result["audit_returncode"] = proc.returncode
        result["audit_stdout"] = proc.stdout
        result["audit_stderr"] = proc.stderr
    return result


def _resolver_command(
    *,
    repo_root: Path,
    pr_number: int,
    gh_bin: str,
    apply: bool,
    expected_merge_commit: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "resolve_lane_conflicts.py"),
        "--merged-pr-lane-audit",
        "--pr",
        str(pr_number),
        "--gh-bin",
        gh_bin,
        "--json",
    ]
    if apply:
        command.extend(
            [
                "--expected-merge-commit",
                str(expected_merge_commit or ""),
                "--operator-authorized",
                "--apply",
            ]
        )
    return command


def _merge_helper_metadata(
    payload: dict[str, Any],
    *,
    audit_ok: bool,
    audit_applied: bool,
    apply_requested: bool,
    command: list[str],
    proc: subprocess.CompletedProcess[str],
    error: str | None = None,
) -> dict[str, Any]:
    result = dict(payload)
    result.update(
        {
            "audit_ok": audit_ok,
            "audit_applied": audit_applied,
            "audit_apply_requested": apply_requested,
            "audit_command": command,
            "audit_returncode": proc.returncode,
        }
    )
    if error:
        result["audit_error"] = error
    return result


def run_post_merge_lane_audit(
    pr_number: int,
    *,
    repo_root: Path | None = None,
    apply: bool = False,
    gh_bin: str = "gh",
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run the merged-PR lane audit and optionally apply guarded cleanup.

    Dry-run mode returns the resolver JSON plus helper metadata and never
    mutates lane state. Apply mode first runs the same dry-run, then reruns the
    resolver with ``--apply --operator-authorized --expected-merge-commit`` only
    when GitHub reports the PR is merged and exposes the live merge commit.
    """

    root = repo_root or REPO_ROOT
    run = runner
    dry_command = _resolver_command(repo_root=root, pr_number=pr_number, gh_bin=gh_bin, apply=False)
    try:
        dry_proc = _run_command(dry_command, cwd=root, runner=run)
    except OSError as exc:
        return _command_error(
            command=dry_command,
            proc=None,
            message=str(exc),
            apply_requested=apply,
        )
    dry_payload = _parse_json(dry_proc.stdout)
    if dry_payload is None:
        return _command_error(
            command=dry_command,
            proc=dry_proc,
            message="merged-PR lane audit returned invalid JSON",
            apply_requested=apply,
        )
    if dry_proc.returncode != 0:
        message = str(dry_payload.get("blocked_reason") or dry_proc.stderr or "audit failed")
        return _merge_helper_metadata(
            dry_payload,
            audit_ok=False,
            audit_applied=False,
            apply_requested=apply,
            command=dry_command,
            proc=dry_proc,
            error=message,
        )
    if dry_payload.get("blocked_reason") in DRY_RUN_ZERO_EXIT_BLOCKERS:
        message = str(dry_payload.get("blocked_reason") or "audit blocked")
        return _merge_helper_metadata(
            dry_payload,
            audit_ok=False,
            audit_applied=False,
            apply_requested=apply,
            command=dry_command,
            proc=dry_proc,
            error=message,
        )

    dry_result = _merge_helper_metadata(
        dry_payload,
        audit_ok=True,
        audit_applied=False,
        apply_requested=apply,
        command=dry_command,
        proc=dry_proc,
    )
    if not apply:
        return dry_result

    github_state = dry_payload.get("github_state")
    if not isinstance(github_state, dict):
        dry_result["audit_ok"] = False
        dry_result["audit_error"] = "github_state_missing"
        return dry_result
    if github_state.get("available") is not True or github_state.get("state") != "MERGED":
        dry_result["audit_ok"] = False
        dry_result["audit_error"] = str(dry_payload.get("blocked_reason") or "pr_not_merged")
        return dry_result
    merge_commit = str(github_state.get("mergeCommit") or "").strip()
    if not merge_commit:
        dry_result["audit_ok"] = False
        dry_result["audit_error"] = "merge_commit_missing"
        return dry_result
    if int(dry_payload.get("finding_count") or 0) == 0:
        dry_result["audit_apply_skipped_reason"] = "no_active_lanes_for_merged_pr"
        return dry_result

    apply_command = _resolver_command(
        repo_root=root,
        pr_number=pr_number,
        gh_bin=gh_bin,
        apply=True,
        expected_merge_commit=merge_commit,
    )
    try:
        apply_proc = _run_command(apply_command, cwd=root, runner=run)
    except OSError as exc:
        return _command_error(
            command=apply_command,
            proc=None,
            message=str(exc),
            apply_requested=True,
        )
    apply_payload = _parse_json(apply_proc.stdout)
    if apply_payload is None:
        return _command_error(
            command=apply_command,
            proc=apply_proc,
            message="merged-PR lane audit apply returned invalid JSON",
            apply_requested=True,
        )
    error = None
    audit_ok = apply_proc.returncode == 0 and not apply_payload.get("blocked_reason")
    if not audit_ok:
        error = str(
            apply_payload.get("blocked_reason") or apply_proc.stderr or "audit apply failed"
        )
    return _merge_helper_metadata(
        apply_payload,
        audit_ok=audit_ok,
        audit_applied=audit_ok,
        apply_requested=True,
        command=apply_command,
        proc=apply_proc,
        error=error,
    )


def post_merge_lane_audit_failed(
    result: Mapping[str, Any] | None,
    *,
    apply_requested: bool,
) -> bool:
    """Return whether an audit result should make the caller fail.

    Dry-run audit failures are surfaced in output but intentionally do not
    reclassify a successful merge. Explicit apply failures are command-failing
    because the operator requested lane mutation and it did not complete.
    """

    if not apply_requested or result is None:
        return False
    return result.get("audit_ok") is False


def post_merge_lane_audit_failure_reason(result: Mapping[str, Any] | None) -> str:
    if result is None:
        return "post-merge lane audit unavailable"
    return str(
        result.get("audit_error")
        or result.get("blocked_reason")
        or result.get("audit_stderr")
        or "post-merge lane audit failed"
    )
