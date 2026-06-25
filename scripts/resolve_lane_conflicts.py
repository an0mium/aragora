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
HEARTBEAT_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "heartbeats.json"
STEERING_INBOX_RELATIVE_DIR = Path(".aragora") / "operator-steering"
HEARTBEAT_FRESH_SECONDS = 15 * 60
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
TERMINAL_PR_STATES = {"MERGED", "CLOSED"}
HEARTBEAT_TIMESTAMP_KEYS = ("last_heartbeat_at", "last_seen_at", "heartbeat_at")
PID_KEYS = ("pid", "owner_pid")
LOCAL_WORK_KEYS = ("worktree", "local_worktree", "local_work_path")


def _canonical_repo_root(path: Path = DEFAULT_REPO_ROOT) -> Path:
    common_dir_proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if common_dir_proc.returncode == 0:
        common_dir = common_dir_proc.stdout.strip()
        if common_dir.endswith("/.git"):
            return Path(common_dir).resolve().parent

    root_proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if root_proc.returncode == 0 and root_proc.stdout.strip():
        return Path(root_proc.stdout.strip()).resolve()
    return path.resolve()


def _git_common_state_root(path: Path = DEFAULT_REPO_ROOT) -> Path | None:
    common_dir_proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if common_dir_proc.returncode != 0 or not common_dir_proc.stdout.strip():
        return None
    common_dir = Path(common_dir_proc.stdout.strip()).resolve()
    if common_dir.name == ".git":
        return common_dir.parent / ".aragora"
    for parent in common_dir.parents:
        if parent.name == ".git":
            return parent.parent / ".aragora"
    return None


def _git_toplevel(path: Path) -> Path | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip()).resolve()


def _registered_worktree_roots(repo_root: Path) -> set[Path]:
    roots: set[Path] = set()
    toplevel = _git_toplevel(repo_root)
    if toplevel is not None:
        roots.add(toplevel)
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return roots
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            roots.add(Path(line.removeprefix("worktree ")).resolve())
    return roots


def _state_root_repo_candidate(state_root: Path) -> Path:
    return state_root.parent if state_root.name == ".aragora" else state_root


def _is_registered_worktree_state_root(state_root: Path, repo_root: Path) -> bool:
    candidate = _state_root_repo_candidate(state_root)
    candidate_root = _git_toplevel(candidate)
    if candidate_root is None or candidate.resolve() != candidate_root:
        return False
    return candidate_root in _registered_worktree_roots(repo_root)


def _trusted_automation_state_roots(repo_root: Path = DEFAULT_REPO_ROOT) -> set[Path]:
    roots = {
        (_canonical_repo_root(repo_root) / ".aragora").resolve(),
        (DEFAULT_REPO_ROOT / ".aragora").resolve(),
    }
    common_state_root = _git_common_state_root(repo_root)
    if common_state_root is not None:
        roots.add(common_state_root.resolve())
    return roots


def _normalize_automation_state_root(path: str) -> Path:
    root = Path(path).expanduser()
    root = root if root.name == ".aragora" else root / ".aragora"
    return root.resolve()


def _automation_state_root(repo_root: Path = DEFAULT_REPO_ROOT) -> Path:
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    if configured:
        root = _normalize_automation_state_root(configured)
        trusted_roots = _trusted_automation_state_roots(repo_root)
        if root not in trusted_roots and not _is_registered_worktree_state_root(root, repo_root):
            allowed = ", ".join(str(item) for item in sorted(trusted_roots))
            raise ValueError(
                f"untrusted ARAGORA_AUTOMATION_STATE_ROOT {root}; expected one of: "
                f"{allowed}, or a registered worktree's .aragora"
            )
        return root
    return (_canonical_repo_root(repo_root) / ".aragora").resolve()


def _validate_gh_bin(gh_bin: str) -> str:
    value = str(gh_bin)
    if value != value.strip() or any(char.isspace() for char in value) or "\0" in value:
        raise ValueError("gh_bin must be one executable token")
    if value == "gh":
        return value
    path = Path(value).expanduser()
    if not path.is_absolute() and not any(sep in value for sep in ("/", os.sep)):
        raise ValueError("gh_bin must be 'gh' or an executable path")
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ValueError(f"gh_bin path could not be resolved: {exc}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("gh_bin path must be an executable file")
    return str(resolved)


def _default_registry_path() -> Path:
    return _automation_state_root() / "agent-bridge" / "lanes.json"


def _default_receipt_dir() -> Path:
    return _automation_state_root() / "agent-bridge" / "conflict-resolution-receipts"


def _default_heartbeat_path() -> Path:
    root = _automation_state_root()
    if root.name == ".aragora":
        return root.joinpath(*HEARTBEAT_RELATIVE_PATH.parts[1:])
    return root / HEARTBEAT_RELATIVE_PATH


def _default_steering_inbox_root() -> Path:
    root = _automation_state_root()
    return (
        root.joinpath(*STEERING_INBOX_RELATIVE_DIR.parts[1:])
        if root.name == ".aragora"
        else root / STEERING_INBOX_RELATIVE_DIR
    )


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


def _read_rows_checked(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"read_failed:{type(exc).__name__}:{exc}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"invalid_json:{exc.msg}"
    if not isinstance(payload, list):
        return [], "invalid_shape:not_list"
    return [row for row in payload if isinstance(row, dict)], None


def _parse_timestamp(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.timestamp()


def _matching_heartbeat_rows(
    heartbeats: list[dict[str, Any]],
    *,
    lane_id: str,
    owner_session: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in heartbeats
        if str(row.get("lane_id") or "") == lane_id
        and str(row.get("owner_session") or "") == owner_session
    ]


def _record_is_terminal(record: dict[str, Any]) -> bool:
    return bool(record.get("terminal") or record.get("terminal_outcome"))


def _fresh_heartbeat_timestamps(
    records: Sequence[dict[str, Any]],
    *,
    now_ts: float,
    freshness_seconds: int,
) -> list[str]:
    fresh: list[str] = []
    cutoff = now_ts - freshness_seconds
    for record in records:
        if _record_is_terminal(record):
            continue
        for key in HEARTBEAT_TIMESTAMP_KEYS:
            raw = record.get(key)
            ts = _parse_timestamp(raw)
            if ts is not None and ts >= cutoff:
                fresh.append(str(raw))
    return fresh


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _live_pids(records: Sequence[dict[str, Any]]) -> list[int]:
    live: list[int] = []
    for record in records:
        if _record_is_terminal(record):
            continue
        for key in PID_KEYS:
            pid = _coerce_int(record.get(key))
            if pid is not None and _process_is_alive(pid):
                live.append(pid)
    return live


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _local_work_claims(records: Sequence[dict[str, Any]]) -> list[str]:
    claims: list[str] = []
    for record in records:
        has_local_work_risk = _truthy_flag(record.get("possible_unpushed_work")) or _truthy_flag(
            record.get("has_unpushed_work")
        )
        if not has_local_work_risk:
            continue
        for key in LOCAL_WORK_KEYS:
            value = str(record.get(key) or "").strip()
            if value:
                claims.append(value)
        if not any(str(record.get(key) or "").strip() for key in LOCAL_WORK_KEYS):
            claims.append("possible_unpushed_work")
    return claims


def _safe_steering_inbox(owner_session: str, *, steering_inbox_root: Path) -> Path | None:
    if (
        not owner_session
        or owner_session != owner_session.strip()
        or "/" in owner_session
        or "\\" in owner_session
    ):
        return None
    session_path = Path(owner_session)
    if session_path.is_absolute() or owner_session in {".", ".."} or ".." in session_path.parts:
        return None
    root = steering_inbox_root.resolve(strict=False)
    inbox = (root / owner_session).resolve(strict=False)
    try:
        inbox.relative_to(root)
    except ValueError:
        return None
    return inbox


def _pending_mailbox_messages(
    owner_session: str, *, steering_inbox_root: Path
) -> tuple[list[str], str | None]:
    inbox = _safe_steering_inbox(owner_session, steering_inbox_root=steering_inbox_root)
    if inbox is None:
        return [], "invalid_owner_session"
    if not inbox.is_dir():
        return [], None
    # Read receipts are proof-of-read/outcome only; they are not an ack/move protocol.
    # A top-level message file remains pending until a future explicit ack path moves it.
    return sorted(path.name for path in inbox.glob("*.json") if path.is_file()), None


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
    try:
        safe_gh_bin = _validate_gh_bin(gh_bin)
    except ValueError as exc:
        return {
            "available": False,
            "state": None,
            "error": f"invalid GitHub CLI configuration: {exc}",
            "command": [],
        }
    cmd = [
        safe_gh_bin,
        "pr",
        "view",
        str(pr),
        "--json",
        "number,state,headRefOid,closedAt,mergedAt,mergeCommit,url",
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
        "closedAt": payload.get("closedAt"),
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
                "local_worktree": row.get("local_worktree"),
                "local_work_path": row.get("local_work_path"),
                "possible_unpushed_work": row.get("possible_unpushed_work"),
                "has_unpushed_work": row.get("has_unpushed_work"),
                "pid": row.get("pid"),
                "owner_pid": row.get("owner_pid"),
                "next_action": row.get("next_action"),
                "updated_at": row.get("updated_at"),
                "last_heartbeat_at": row.get("last_heartbeat_at"),
                "last_steering_outcome": row.get("last_steering_outcome"),
                "pr_number": pr,
            }
        )
    return findings


def _annotate_terminal_safety(
    findings: list[dict[str, Any]],
    *,
    heartbeats: list[dict[str, Any]],
    heartbeat_load_error: str | None = None,
    steering_inbox_root: Path,
    now_ts: float,
    heartbeat_fresh_seconds: int,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for finding in findings:
        finding = dict(finding)
        lane_id = str(finding.get("lane_id") or "")
        owner_session = str(finding.get("owner_session") or "")
        heartbeat_rows = _matching_heartbeat_rows(
            heartbeats,
            lane_id=lane_id,
            owner_session=owner_session,
        )
        records = [finding, *heartbeat_rows]
        blockers: list[str] = []
        details: dict[str, Any] = {}

        live_pids = sorted(set(_live_pids(records)))
        if live_pids:
            blockers.append("live_process")
            details["live_pids"] = live_pids

        fresh_heartbeats = _fresh_heartbeat_timestamps(
            records,
            now_ts=now_ts,
            freshness_seconds=heartbeat_fresh_seconds,
        )
        if fresh_heartbeats:
            blockers.append("fresh_heartbeat")
            details["fresh_heartbeat_timestamps"] = fresh_heartbeats

        if heartbeat_load_error:
            blockers.append("heartbeat_state_untrusted")
            details["heartbeat_read_error"] = heartbeat_load_error

        pending_messages, mailbox_blocker = _pending_mailbox_messages(
            owner_session,
            steering_inbox_root=steering_inbox_root,
        )
        if mailbox_blocker:
            blockers.append(mailbox_blocker)
        elif pending_messages:
            blockers.append("unread_mailbox")
            details["pending_mailbox_messages"] = pending_messages

        local_work_claims = sorted(set(_local_work_claims(records)))
        if local_work_claims:
            blockers.append("local_work_claim")
            details["local_work_claims"] = local_work_claims

        finding["terminal_safety_blockers"] = blockers
        finding["terminal_safety_details"] = details
        finding["apply_safe"] = not blockers
        annotated.append(finding)
    return annotated


def _quote_shell_arg(value: Any) -> str:
    text = str(value or "")
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _terminal_pr_state(github_state: dict[str, Any]) -> str:
    return str(github_state.get("state") or "").upper()


def _terminal_pr_proof_matches(
    *,
    github_state: dict[str, Any],
    expected_merge_commit: str | None,
    expected_closed_at: str | None,
    expected_head_sha: str | None,
) -> bool:
    state = _terminal_pr_state(github_state)
    if state == "MERGED":
        expected = str(expected_merge_commit or "")
        return bool(expected) and str(github_state.get("mergeCommit") or "") == expected
    if state == "CLOSED":
        expected_closed = str(expected_closed_at or "")
        expected_head = str(expected_head_sha or "")
        return (
            bool(expected_closed)
            and bool(expected_head)
            and str(github_state.get("closedAt") or "") == expected_closed
            and str(github_state.get("headRefOid") or "") == expected_head
        )
    return False


def _steering_body(*, pr: int, github_state: dict[str, Any]) -> str:
    merge_commit = github_state.get("mergeCommit") or ""
    head = github_state.get("headRefOid") or ""
    merged_at = github_state.get("mergedAt") or ""
    closed_at = github_state.get("closedAt") or ""
    state = _terminal_pr_state(github_state)
    if state == "CLOSED":
        terminal = f"closed at {closed_at} from head {head}"
        target = "already-closed target"
    else:
        terminal = f"merged at {merge_commit} from head {head} (merged_at={merged_at})"
        target = "already-merged target"
    return (
        f"PR #{pr} is already {terminal}. Please mark this lane completed, released, "
        "or superseded via claim_active_agent_lane.py; do not continue PR mutation work "
        f"on this {target}."
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
    closed_at = github_state.get("closedAt") or ""
    state = _terminal_pr_state(github_state)
    terminal = f"closed at {closed_at}" if state == "CLOSED" else f"merged at {merge_commit}"
    next_action = _quote_shell_arg(f"superseded after PR #{pr} {terminal}; no further PR mutation")
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
    expected_closed_at: str,
    expected_head_sha: str,
    heartbeat_path: Path,
    steering_inbox_root: Path,
    heartbeat_fresh_seconds: int,
) -> str:
    terminal_guard = (
        (
            f"--expected-closed-at {expected_closed_at or '<closed-at>'} "
            f"--expected-head-sha {expected_head_sha or '<head-sha>'}"
        )
        if expected_closed_at
        else f"--expected-merge-commit {expected_merge_commit or '<merge-commit-sha>'}"
    )
    return (
        "python3 scripts/resolve_lane_conflicts.py --merged-pr-lane-audit "
        f"--pr {pr} {terminal_guard} --operator-authorized "
        f"--registry-path {registry_path} --receipt-dir {receipt_dir} "
        f"--heartbeat-path {_quote_shell_arg(heartbeat_path)} "
        f"--steering-inbox-root {_quote_shell_arg(steering_inbox_root)} "
        f"--heartbeat-fresh-seconds {heartbeat_fresh_seconds} --apply --json"
    )


def _base_merged_pr_audit_result(
    *,
    registry_path: Path,
    receipt_dir: Path,
    pr: int,
    apply: bool,
    operator_authorized: bool,
    expected_merge_commit: str | None,
    expected_closed_at: str | None,
    expected_head_sha: str | None,
    github_state: dict[str, Any],
    findings: list[dict[str, Any]],
    blocked_reason: str | None,
    heartbeat_path: Path,
    steering_inbox_root: Path,
    heartbeat_fresh_seconds: int,
) -> dict[str, Any]:
    merge_commit = str(github_state.get("mergeCommit") or "")
    closed_at = str(github_state.get("closedAt") or "")
    head_sha = str(github_state.get("headRefOid") or "")
    expected = str(expected_merge_commit or "")
    expected_closed = str(expected_closed_at or "")
    expected_head = str(expected_head_sha or "")
    unsafe_findings = [finding for finding in findings if finding.get("terminal_safety_blockers")]
    safe_findings = [finding for finding in findings if not finding.get("terminal_safety_blockers")]
    apply_eligible = (
        apply
        and operator_authorized
        and bool(safe_findings)
        and github_state.get("available") is True
        and _terminal_pr_state(github_state) in TERMINAL_PR_STATES
        and _terminal_pr_proof_matches(
            github_state=github_state,
            expected_merge_commit=expected,
            expected_closed_at=expected_closed,
            expected_head_sha=expected_head,
        )
    )
    operator_apply_command = ""
    if heartbeat_fresh_seconds >= 0:
        terminal_state = _terminal_pr_state(github_state)
        operator_apply_command = _operator_apply_command(
            pr=pr,
            registry_path=registry_path,
            receipt_dir=receipt_dir,
            expected_merge_commit=merge_commit or expected,
            expected_closed_at=(
                closed_at or expected_closed or "<closed-at>" if terminal_state == "CLOSED" else ""
            ),
            expected_head_sha=(
                head_sha or expected_head or "<head-sha>" if terminal_state == "CLOSED" else ""
            ),
            heartbeat_path=heartbeat_path,
            steering_inbox_root=steering_inbox_root,
            heartbeat_fresh_seconds=heartbeat_fresh_seconds,
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
        "safe_finding_count": len(findings) - len(unsafe_findings),
        "unsafe_finding_count": len(unsafe_findings),
        "owner_steering_text": "\n".join(
            _owner_steering_commands(findings=findings, pr=pr, github_state=github_state)
        ),
        "owner_release_commands": _owner_release_commands(
            findings=findings,
            pr=pr,
            github_state=github_state,
        ),
        "operator_apply_command": operator_apply_command,
        "requires_operator_authorization": True,
        "operator_authorized": operator_authorized,
        "expected_merge_commit": expected,
        "expected_closed_at": expected_closed,
        "expected_head_sha": expected_head,
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
    expected_closed_at: str | None,
    expected_head_sha: str | None,
    github_state: dict[str, Any],
    findings: list[dict[str, Any]],
) -> str | None:
    if github_state.get("available") is not True:
        return "github_state_unavailable"
    state = _terminal_pr_state(github_state)
    if state not in TERMINAL_PR_STATES:
        return "pr_not_merged"
    if not findings:
        return "no_active_lanes_for_merged_pr"
    if not apply:
        return None
    if not operator_authorized:
        return "operator_authorization_required"
    if state == "MERGED":
        expected = str(expected_merge_commit or "")
        if not expected:
            return "expected_merge_commit_required"
        if str(github_state.get("mergeCommit") or "") != expected:
            return "merge_commit_mismatch"
    if state == "CLOSED":
        expected_closed = str(expected_closed_at or "")
        expected_head = str(expected_head_sha or "")
        if not expected_closed:
            return "expected_closed_at_required"
        if str(github_state.get("closedAt") or "") != expected_closed:
            return "closed_at_mismatch"
        if not expected_head:
            return "expected_head_sha_required"
        if str(github_state.get("headRefOid") or "") != expected_head:
            return "head_sha_mismatch"
    unsafe_findings = [finding for finding in findings if finding.get("terminal_safety_blockers")]
    safe_findings = [finding for finding in findings if not finding.get("terminal_safety_blockers")]
    if unsafe_findings and not safe_findings:
        return "unsafe_terminal_owner_gates"
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
    expected_closed_at: str | None = None,
    expected_head_sha: str | None = None,
    resolved_at: str | None = None,
    heartbeat_path: Path | None = None,
    steering_inbox_root: Path | None = None,
    heartbeat_fresh_seconds: int = HEARTBEAT_FRESH_SECONDS,
) -> dict[str, Any]:
    """Audit active/blocked lane rows for an already-terminal PR.

    Dry-run mode never mutates. Apply mode requires explicit operator
    authorization and exact terminal proof before superseding active lifecycle rows.
    """

    resolved_at = resolved_at or _utc_now_iso()
    try:
        heartbeat_path = heartbeat_path or _default_heartbeat_path()
        steering_inbox_root = steering_inbox_root or _default_steering_inbox_root()
    except ValueError as exc:
        return {
            "mode": "merged_pr_lane_audit",
            "apply": bool(apply),
            "apply_eligible": False,
            "blocked_reason": "invalid_automation_state_root",
            "error": str(exc),
            "finding_count": 0,
            "findings": [],
            "github_state": {"available": False, "error": str(exc)},
            "heartbeat_load_error": None,
            "owner_release_commands": [],
            "owner_steering_commands": [],
            "owner_steering_text": "",
            "receipt_paths": [],
            "resolved_count": 0,
        }
    now_ts = _parse_timestamp(resolved_at) or dt.datetime.now(dt.UTC).timestamp()
    github_state = _fetch_pr_state(pr=pr, gh_bin=gh_bin)
    with _registry_write_lock(registry_path):
        rows = _read_rows(registry_path)
        heartbeats, heartbeat_load_error = _read_rows_checked(heartbeat_path)
        findings: list[dict[str, Any]] = []
        if (
            github_state.get("available") is True
            and _terminal_pr_state(github_state) in TERMINAL_PR_STATES
        ):
            raw_findings = _active_pr_lane_findings(rows, pr=pr)
            if heartbeat_fresh_seconds < 0:
                findings = []
                for finding in raw_findings:
                    finding = dict(finding)
                    finding["terminal_safety_blockers"] = ["invalid_heartbeat_fresh_seconds"]
                    finding["terminal_safety_details"] = {
                        "heartbeat_fresh_seconds": heartbeat_fresh_seconds
                    }
                    finding["apply_safe"] = False
                    findings.append(finding)
            else:
                findings = _annotate_terminal_safety(
                    raw_findings,
                    heartbeats=heartbeats,
                    heartbeat_load_error=heartbeat_load_error,
                    steering_inbox_root=steering_inbox_root,
                    now_ts=now_ts,
                    heartbeat_fresh_seconds=heartbeat_fresh_seconds,
                )
        blocked_reason = _merged_pr_audit_blocked_reason(
            apply=apply,
            operator_authorized=operator_authorized,
            expected_merge_commit=expected_merge_commit,
            expected_closed_at=expected_closed_at,
            expected_head_sha=expected_head_sha,
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
            expected_closed_at=expected_closed_at,
            expected_head_sha=expected_head_sha,
            github_state=github_state,
            findings=findings,
            blocked_reason=blocked_reason,
            heartbeat_path=heartbeat_path,
            steering_inbox_root=steering_inbox_root,
            heartbeat_fresh_seconds=heartbeat_fresh_seconds,
        )
        if not result["apply_eligible"]:
            if heartbeat_fresh_seconds < 0 and findings:
                result["blocked_reason"] = "invalid_heartbeat_fresh_seconds"
            return result

        safe_findings = [
            finding for finding in findings if not finding.get("terminal_safety_blockers")
        ]
        target_keys = {
            (str(finding.get("lane_id") or ""), str(finding.get("owner_session") or ""))
            for finding in safe_findings
        }
        findings_by_key = {
            (str(finding.get("lane_id") or ""), str(finding.get("owner_session") or "")): finding
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
                terminal_state = _terminal_pr_state(github_state)
                receipt = {
                    "schema_version": MERGED_PR_RECEIPT_SCHEMA_VERSION,
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "pr_number": pr,
                    "head_sha": github_state.get("headRefOid"),
                    "terminal_state": terminal_state,
                    "merge_commit": github_state.get("mergeCommit"),
                    "closed_at": github_state.get("closedAt"),
                    "merged_at": github_state.get("mergedAt"),
                    "old_status": old_status,
                    "new_status": "superseded",
                    "resolved_at_utc": resolved_at,
                    "resolution": (
                        "closed_pr_has_active_lane_row"
                        if terminal_state == "CLOSED"
                        else "merged_pr_has_active_lane_row"
                    ),
                    "terminal_safety_blockers": findings_by_key.get(row_key, {}).get(
                        "terminal_safety_blockers",
                        [],
                    ),
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
        "--expected-closed-at",
        default="",
        help="Exact closedAt timestamp required for authorized closed-PR apply mode.",
    )
    parser.add_argument(
        "--expected-head-sha",
        default="",
        help="Exact headRefOid required for authorized closed-PR apply mode.",
    )
    parser.add_argument(
        "--operator-authorized",
        action="store_true",
        help="Required with --apply in merged-PR lane audit mode.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help=(
            "Lane registry path. Defaults to ARAGORA_AUTOMATION_STATE_ROOT, then "
            "the canonical repo root's .aragora/agent-bridge/lanes.json."
        ),
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=None,
        help=(
            "Directory for append-only resolution receipts. Defaults to the "
            "shared automation state root."
        ),
    )
    parser.add_argument(
        "--heartbeat-path",
        type=Path,
        default=None,
        help="Override .aragora/agent-bridge/heartbeats.json for terminal safety checks.",
    )
    parser.add_argument(
        "--steering-inbox-root",
        type=Path,
        default=None,
        help="Override .aragora/operator-steering for unread mailbox checks.",
    )
    parser.add_argument(
        "--heartbeat-fresh-seconds",
        type=int,
        default=HEARTBEAT_FRESH_SECONDS,
        help="Fresh heartbeat TTL in seconds for merged-PR apply safety checks.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry_path = args.registry_path or _default_registry_path()
        receipt_dir = args.receipt_dir or _default_receipt_dir()
    except ValueError as exc:
        result = {
            "blocked_reason": "invalid_automation_state_root",
            "candidate_count": 0,
            "error": str(exc),
            "receipt_dir": None,
            "registry_path": None,
            "resolved_count": 0,
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"blocked=invalid_automation_state_root: {exc}")
        return 2
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
            registry_path=registry_path,
            receipt_dir=receipt_dir,
            pr=args.pr,
            gh_bin=args.gh_bin,
            apply=bool(args.apply),
            operator_authorized=bool(args.operator_authorized),
            expected_merge_commit=args.expected_merge_commit,
            expected_closed_at=args.expected_closed_at,
            expected_head_sha=args.expected_head_sha,
            heartbeat_path=args.heartbeat_path,
            steering_inbox_root=args.steering_inbox_root,
            heartbeat_fresh_seconds=args.heartbeat_fresh_seconds,
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
        if result.get("blocked_reason") == "invalid_automation_state_root":
            return 2
        if args.apply and result.get("resolved_count", 0) == 0:
            return 2
        return 0

    result = resolve_conflicts(
        registry_path=registry_path,
        receipt_dir=receipt_dir,
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
