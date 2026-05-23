#!/usr/bin/env python3
"""Resolve stale lane conflict rows with append-only receipts.

The resolver only handles the safe case where a ``status=conflict`` row points
at an owner session that no longer has an active lane. It never deletes rows;
``--apply`` marks the conflict row ``superseded`` and writes a sidecar receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
RECEIPT_RELATIVE_DIR = Path(".aragora") / "agent-bridge" / "conflict-resolution-receipts"
RECEIPT_SCHEMA_VERSION = "aragora-lane-conflict-resolution/1.0"
MERGED_PR_RECEIPT_SCHEMA_VERSION = "aragora-merged-pr-lane-audit/1.0"
ACTIVE_STATUSES = {
    "active",
    "running",
    "pending",
    "queued",
    "claimed",
    "waiting_for_steering",
    "acknowledged",
    "working",
    "blocked",
}
INACTIVE_OWNER_STATUSES = {"released", "completed", "superseded"}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def _registry_write_lock(path: Path) -> Iterator[None]:
    """Serialize conflict-resolution registry writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)


def _owner_is_inactive(rows: list[dict[str, Any]], owner_session: str) -> bool:
    owner_rows = [row for row in rows if str(row.get("owner_session") or "") == owner_session]
    if not owner_rows:
        return False
    return all(str(row.get("status") or "") in INACTIVE_OWNER_STATUSES for row in owner_rows)


def _unknown_conflict_sessions_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_sessions = {str(row.get("owner_session") or "") for row in rows}
    unknown: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") != "conflict":
            continue
        conflict_session = str(row.get("conflict_session") or "")
        if conflict_session and conflict_session not in known_sessions:
            unknown.append(
                {
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "conflict_session": conflict_session,
                    "conflict_reason": row.get("conflict_reason"),
                    "current_status": row.get("status"),
                    "resolution": "requires_manual_review_unknown_conflict_session",
                }
            )
    return unknown


def _find_resolvable_conflicts_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") != "conflict":
            continue
        conflict_session = str(row.get("conflict_session") or "")
        if not conflict_session:
            continue
        if _owner_is_inactive(rows, conflict_session):
            candidates.append(
                {
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "conflict_session": conflict_session,
                    "conflict_reason": row.get("conflict_reason"),
                    "current_status": row.get("status"),
                    "new_status": "superseded",
                    "resolution": "conflict_session_has_only_inactive_rows",
                }
            )
    return candidates


def find_resolvable_conflicts(registry_path: Path) -> list[dict[str, Any]]:
    return _find_resolvable_conflicts_from_rows(_read_rows(registry_path))


def _write_receipt(
    *,
    receipt_dir: Path,
    receipt: dict[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = str(receipt["resolved_at_utc"]).replace(":", "-")
    path = receipt_dir / f"{ts}-{secrets.token_hex(4)}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(receipt_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_commit_oid(payload: dict[str, Any]) -> str:
    merge_commit = payload.get("mergeCommit")
    if isinstance(merge_commit, dict):
        return str(merge_commit.get("oid") or "").strip()
    return ""


def _fetch_pr_state(*, pr: int, gh_bin: str) -> dict[str, Any]:
    cmd = [
        gh_bin,
        "pr",
        "view",
        str(pr),
        "--json",
        "number,state,headRefOid,mergedAt,mergeCommit,url",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "state": None,
            "error": str(exc),
            "command": cmd,
        }
    if proc.returncode != 0:
        return {
            "available": False,
            "state": None,
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "returncode": proc.returncode,
            "command": cmd,
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "state": None,
            "error": f"invalid gh json: {exc}",
            "command": cmd,
        }
    if not isinstance(payload, dict):
        return {
            "available": False,
            "state": None,
            "error": "gh pr view returned non-object JSON",
            "command": cmd,
        }
    return {
        "available": True,
        "number": _coerce_int(payload.get("number")),
        "state": str(payload.get("state") or "").upper(),
        "headRefOid": str(payload.get("headRefOid") or ""),
        "mergedAt": payload.get("mergedAt"),
        "mergeCommit": _merge_commit_oid(payload),
        "url": str(payload.get("url") or ""),
        "command": cmd,
    }


def _active_pr_lane_findings(rows: list[dict[str, Any]], *, pr: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for row in rows:
        if _coerce_int(row.get("pr_number")) != pr:
            continue
        status = str(row.get("status") or "")
        if status not in ACTIVE_STATUSES:
            continue
        findings.append(
            {
                "lane_id": row.get("lane_id"),
                "owner_session": row.get("owner_session"),
                "status": status,
                "branch": row.get("branch"),
                "worktree": row.get("worktree"),
                "next_action": row.get("next_action"),
                "updated_at": row.get("updated_at"),
                "last_heartbeat_at": row.get("last_heartbeat_at"),
                "last_steering_outcome": row.get("last_steering_outcome"),
                "pr_number": pr,
            }
        )
    return findings


def _quote_shell_arg(value: Any) -> str:
    text = str(value or "")
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _steering_body(*, pr: int, github_state: dict[str, Any]) -> str:
    merge_commit = github_state.get("mergeCommit") or ""
    head = github_state.get("headRefOid") or ""
    merged_at = github_state.get("mergedAt") or ""
    return (
        f"PR #{pr} is already merged at {merge_commit} from head {head} "
        f"(merged_at={merged_at}). Please mark this lane completed, released, "
        "or superseded via claim_active_agent_lane.py; do not continue PR "
        "mutation work on this already-merged target."
    )


def _owner_steering_commands(
    *,
    findings: list[dict[str, Any]],
    pr: int,
    github_state: dict[str, Any],
) -> list[str]:
    body = _quote_shell_arg(_steering_body(pr=pr, github_state=github_state))
    commands: list[str] = []
    for finding in findings:
        owner = finding.get("owner_session")
        lane_id = finding.get("lane_id")
        if not owner:
            continue
        cmd = (
            f"python3 scripts/send_operator_steering.py --to {owner} "
            f"--pr {pr} --priority blocking --body {body}"
        )
        if lane_id:
            cmd += f" --lane-id {lane_id}"
        commands.append(cmd)
    return commands


def _owner_release_commands(
    *,
    findings: list[dict[str, Any]],
    pr: int,
    github_state: dict[str, Any],
) -> list[str]:
    merge_commit = github_state.get("mergeCommit") or ""
    next_action = _quote_shell_arg(
        f"superseded after PR #{pr} merged at {merge_commit}; no further PR mutation"
    )
    commands: list[str] = []
    for finding in findings:
        lane_id = finding.get("lane_id")
        owner = finding.get("owner_session")
        if not lane_id or not owner:
            continue
        commands.append(
            "python3 scripts/claim_active_agent_lane.py "
            f"--lane-id {lane_id} --owner-session {owner} "
            f"--status superseded --pr-number {pr} "
            f"--next-action {next_action} --json"
        )
    return commands


def _operator_apply_command(
    *,
    pr: int,
    registry_path: Path,
    receipt_dir: Path,
    expected_merge_commit: str,
) -> str:
    commit_arg = expected_merge_commit or "<merge-commit-sha>"
    return (
        "python3 scripts/resolve_lane_conflicts.py --merged-pr-lane-audit "
        f"--pr {pr} --expected-merge-commit {commit_arg} --operator-authorized "
        f"--registry-path {registry_path} --receipt-dir {receipt_dir} --apply --json"
    )


def _base_merged_pr_audit_result(
    *,
    registry_path: Path,
    receipt_dir: Path,
    pr: int,
    apply: bool,
    operator_authorized: bool,
    expected_merge_commit: str | None,
    github_state: dict[str, Any],
    findings: list[dict[str, Any]],
    blocked_reason: str | None,
) -> dict[str, Any]:
    merge_commit = str(github_state.get("mergeCommit") or "")
    expected = str(expected_merge_commit or "")
    apply_eligible = (
        apply
        and operator_authorized
        and bool(expected)
        and bool(findings)
        and github_state.get("available") is True
        and github_state.get("state") == "MERGED"
        and merge_commit == expected
    )
    return {
        "mode": "merged_pr_lane_audit",
        "registry_path": str(registry_path),
        "receipt_dir": str(receipt_dir),
        "dry_run": not apply,
        "pr_number": pr,
        "github_state": github_state,
        "finding_count": len(findings),
        "findings": findings,
        "owner_steering_text": "\n".join(
            _owner_steering_commands(findings=findings, pr=pr, github_state=github_state)
        ),
        "owner_release_commands": _owner_release_commands(
            findings=findings,
            pr=pr,
            github_state=github_state,
        ),
        "operator_apply_command": _operator_apply_command(
            pr=pr,
            registry_path=registry_path,
            receipt_dir=receipt_dir,
            expected_merge_commit=merge_commit or expected,
        ),
        "requires_operator_authorization": True,
        "operator_authorized": operator_authorized,
        "expected_merge_commit": expected,
        "apply_eligible": apply_eligible,
        "blocked_reason": blocked_reason,
        "resolved_count": 0,
        "receipt_paths": [],
    }


def _merged_pr_audit_blocked_reason(
    *,
    apply: bool,
    operator_authorized: bool,
    expected_merge_commit: str | None,
    github_state: dict[str, Any],
    findings: list[dict[str, Any]],
) -> str | None:
    if github_state.get("available") is not True:
        return "github_state_unavailable"
    if github_state.get("state") != "MERGED":
        return "pr_not_merged"
    if not findings:
        return "no_active_lanes_for_merged_pr"
    if not apply:
        return None
    if not operator_authorized:
        return "operator_authorization_required"
    expected = str(expected_merge_commit or "")
    if not expected:
        return "expected_merge_commit_required"
    if str(github_state.get("mergeCommit") or "") != expected:
        return "merge_commit_mismatch"
    return None


def audit_merged_pr_lanes(
    *,
    registry_path: Path,
    receipt_dir: Path,
    pr: int,
    gh_bin: str = "gh",
    apply: bool = False,
    operator_authorized: bool = False,
    expected_merge_commit: str | None = None,
    resolved_at: str | None = None,
) -> dict[str, Any]:
    """Audit active/blocked lane rows for an already-merged PR.

    Dry-run mode never mutates. Apply mode requires explicit operator
    authorization and an exact merge-commit guard before superseding active
    lifecycle rows.
    """

    resolved_at = resolved_at or _utc_now_iso()
    github_state = _fetch_pr_state(pr=pr, gh_bin=gh_bin)
    with _registry_write_lock(registry_path):
        rows = _read_rows(registry_path)
        findings: list[dict[str, Any]] = []
        if github_state.get("available") is True and github_state.get("state") == "MERGED":
            findings = _active_pr_lane_findings(rows, pr=pr)
        blocked_reason = _merged_pr_audit_blocked_reason(
            apply=apply,
            operator_authorized=operator_authorized,
            expected_merge_commit=expected_merge_commit,
            github_state=github_state,
            findings=findings,
        )
        result = _base_merged_pr_audit_result(
            registry_path=registry_path,
            receipt_dir=receipt_dir,
            pr=pr,
            apply=apply,
            operator_authorized=operator_authorized,
            expected_merge_commit=expected_merge_commit,
            github_state=github_state,
            findings=findings,
            blocked_reason=blocked_reason,
        )
        if not result["apply_eligible"]:
            return result

        target_keys = {
            (str(finding.get("lane_id") or ""), str(finding.get("owner_session") or ""))
            for finding in findings
        }
        receipt_paths: list[str] = []
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            row_key = (str(row.get("lane_id") or ""), str(row.get("owner_session") or ""))
            row_pr = _coerce_int(row.get("pr_number"))
            old_status = str(row.get("status") or "")
            if row_key in target_keys and row_pr == pr and old_status in ACTIVE_STATUSES:
                row["status"] = "superseded"
                row["updated_at"] = resolved_at
                row["last_steering_outcome"] = "superseded"
                receipt = {
                    "schema_version": MERGED_PR_RECEIPT_SCHEMA_VERSION,
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "pr_number": pr,
                    "head_sha": github_state.get("headRefOid"),
                    "merge_commit": github_state.get("mergeCommit"),
                    "merged_at": github_state.get("mergedAt"),
                    "old_status": old_status,
                    "new_status": "superseded",
                    "resolved_at_utc": resolved_at,
                    "resolution": "merged_pr_has_active_lane_row",
                }
                receipt_paths.append(str(_write_receipt(receipt_dir=receipt_dir, receipt=receipt)))
            out_rows.append(row)
        _atomic_write(registry_path, out_rows)
        result["resolved_count"] = len(receipt_paths)
        result["receipt_paths"] = receipt_paths
        result["blocked_reason"] = None
        return result


def resolve_conflicts(
    *,
    registry_path: Path,
    receipt_dir: Path,
    apply: bool = False,
    resolved_at: str | None = None,
) -> dict[str, Any]:
    resolved_at = resolved_at or _utc_now_iso()
    receipt_paths: list[str] = []
    with _registry_write_lock(registry_path):
        rows = _read_rows(registry_path)
        candidates = _find_resolvable_conflicts_from_rows(rows)
        unknown_conflicts = _unknown_conflict_sessions_from_rows(rows)
        if apply and candidates:
            candidate_keys = {
                (
                    str(candidate.get("lane_id") or ""),
                    str(candidate.get("owner_session") or ""),
                    str(candidate.get("conflict_session") or ""),
                )
                for candidate in candidates
            }
            out_rows: list[dict[str, Any]] = []
            for row in rows:
                row = dict(row)
                row_key = (
                    str(row.get("lane_id") or ""),
                    str(row.get("owner_session") or ""),
                    str(row.get("conflict_session") or ""),
                )
                if row_key in candidate_keys and row.get("status") == "conflict":
                    row["status"] = "superseded"
                    row["updated_at"] = resolved_at
                    row["last_steering_outcome"] = "superseded"
                    receipt = {
                        "schema_version": RECEIPT_SCHEMA_VERSION,
                        "lane_id": row.get("lane_id"),
                        "owner_session": row.get("owner_session"),
                        "conflict_session": row.get("conflict_session"),
                        "conflict_reason": row.get("conflict_reason"),
                        "old_status": "conflict",
                        "new_status": "superseded",
                        "resolved_at_utc": resolved_at,
                        "resolution": "conflict_session_has_only_inactive_rows",
                    }
                    receipt_paths.append(
                        str(_write_receipt(receipt_dir=receipt_dir, receipt=receipt))
                    )
                out_rows.append(row)
            _atomic_write(registry_path, out_rows)

    return {
        "registry_path": str(registry_path),
        "dry_run": not apply,
        "resolved_count": len(candidates) if apply else 0,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "unknown_session_count": len(unknown_conflicts),
        "candidates_unknown": unknown_conflicts,
        "receipt_paths": receipt_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", default=True)
    action.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--merged-pr-lane-audit",
        action="store_true",
        help="Audit active/blocked lane rows for an already-merged PR.",
    )
    parser.add_argument("--pr", type=int, help="PR number for --merged-pr-lane-audit.")
    parser.add_argument(
        "--gh-bin",
        default="gh",
        help="GitHub CLI executable for read-only PR state lookup.",
    )
    parser.add_argument(
        "--expected-merge-commit",
        default="",
        help="Exact merge commit required for authorized merged-PR apply mode.",
    )
    parser.add_argument(
        "--operator-authorized",
        action="store_true",
        help="Required with --apply in merged-PR lane audit mode.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REPO_ROOT / REGISTRY_RELATIVE_PATH,
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=DEFAULT_REPO_ROOT / RECEIPT_RELATIVE_DIR,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.merged_pr_lane_audit:
        if args.pr is None:
            result = {
                "mode": "merged_pr_lane_audit",
                "blocked_reason": "pr_required",
                "resolved_count": 0,
                "receipt_paths": [],
            }
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print("blocked: --pr is required for --merged-pr-lane-audit")
            return 2
        result = audit_merged_pr_lanes(
            registry_path=args.registry_path,
            receipt_dir=args.receipt_dir,
            pr=args.pr,
            gh_bin=args.gh_bin,
            apply=bool(args.apply),
            operator_authorized=bool(args.operator_authorized),
            expected_merge_commit=args.expected_merge_commit,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            if result.get("blocked_reason"):
                print(f"blocked={result['blocked_reason']}")
            print(f"finding_count={result['finding_count']}")
            if result.get("owner_steering_text"):
                print(result["owner_steering_text"])
            owner_release_commands = result.get("owner_release_commands")
            if isinstance(owner_release_commands, list):
                for command in owner_release_commands:
                    print(command)
            if result.get("operator_apply_command"):
                print(result["operator_apply_command"])
        if args.apply and result.get("resolved_count", 0) == 0:
            return 2
        return 0

    result = resolve_conflicts(
        registry_path=args.registry_path,
        receipt_dir=args.receipt_dir,
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "resolved" if args.apply else "candidate"
        print(f"{verb}_count={result['resolved_count' if args.apply else 'candidate_count']}")
        candidates = result.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                print(
                    f"- lane_id={candidate['lane_id']} conflict_session="
                    f"{candidate['conflict_session']} -> superseded"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
