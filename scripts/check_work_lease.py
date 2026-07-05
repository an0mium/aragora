#!/usr/bin/env python3
"""Lease-rule preflight: no automated branch push without holding the work lease.

Issue #8851 (acceptance item 2). Five fleets (Factory mission workers,
agent-bridge lanes, boss-loop, queue-drain, conductor sessions) each track
ownership in different systems, producing foreign-commit contamination and
duplicate work. This preflight makes ``aragora.nomic.dev_coordination`` the
single ownership truth: before pushing a branch, a session must hold (or
claim) the active :class:`WorkLease` whose ``branch`` matches.

Design goals:

- **Fast** (<1s): the read-only check uses stdlib ``sqlite3`` directly
  against the live store; the heavyweight ``aragora`` import happens only
  on mutation paths (``--claim`` / ``--renew`` / ``--release``).
- **Fail-open with noise** (v0): if the store is unreachable the check
  prints a WARNING and exits 0 so no fleet is bricked. ``--strict``
  switches to fail-closed (v1 posture).
- **Compose, don't rewrite**: mutations go through
  ``DevCoordinationStore.claim_lease`` / ``heartbeat_lease`` /
  ``release_lease`` so conflict detection, fleet-claim mirroring, and
  event publication keep working.

Exit codes:

- 0: invoking session holds (or just claimed/renewed/released) the lease,
  or the store is unreachable without ``--strict`` (fail-open).
- 1: another session holds the lease, no lease is held and ``--claim`` was
  not given, or the store is unreachable with ``--strict``.
- 2: usage error.

Examples::

    # Preflight before push (claims the lease if free):
    python3 scripts/check_work_lease.py my-branch --claim \
        --session-id "$ARAGORA_SESSION_ID" --agent claude

    # Renew while working; release when done:
    python3 scripts/check_work_lease.py my-branch --renew
    python3 scripts/check_work_lease.py my-branch --release

    # agent-bridge lane adapter (records lease id in the sidecar):
    python3 scripts/check_work_lease.py my-branch --claim --record-lane lane-42

See ``docs/coordination/LEASE_RULE.md`` for the rollout plan.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import getpass
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_ENV_VAR = "ARAGORA_DEV_COORDINATION_DB"
SESSION_ENV_VARS = (
    "ARAGORA_SESSION_ID",
    "ARAGORA_AGENT_SESSION_ID",
    "ARAGORA_SWARM_SESSION_ID",
)
AGENT_ENV_VARS = ("ARAGORA_AGENT", "ARAGORA_AGENT_NAME")
LANE_LEASES_RELPATH = Path(".aragora") / "agent-bridge" / "lane-leases.json"
DEFAULT_TTL_HOURS = 8.0
# Synthetic write-scope entry carried by every helper-mediated claim. The
# store's conflict detection is file-scope based (allowed_globs /
# claimed_paths), so an empty-scope lease would never conflict with another
# empty-scope lease for the same branch. Two identical literal globs DO
# conflict (_glob_overlap: a == b), so this lock path makes any two helper
# claims for the same branch conflict transactionally at the store.
BRANCH_LOCK_GLOB_TEMPLATE = ".aragora/branch-locks/{branch}"

REASON_MISSING_LEASE = "missing_lease"
REASON_EXPIRED_LEASE = "expired_lease"
REASON_WRONG_OWNER = "wrong_owner"
REASON_BRIDGE_ONLY_NO_DEV_LEASE = "bridge_only_no_dev_lease"
REASON_AMBIGUOUS_OWNER = "ambiguous_owner"
REASON_STORE_UNREACHABLE = "store_unreachable"


class StoreUnreachableError(RuntimeError):
    """The dev_coordination store cannot be read or resolved."""


@dataclass(slots=True)
class LeaseRow:
    """Minimal projection of a ``leases`` row for the read-only path."""

    lease_id: str
    task_id: str
    title: str
    owner_agent: str
    owner_session_id: str
    branch: str
    worktree_path: str
    status: str
    created_at: str
    expires_at: str
    metadata: dict[str, Any]

    @property
    def is_expired(self) -> bool:
        return _parse_dt(self.expires_at) <= _utcnow()

    @property
    def work_id(self) -> str | None:
        value = self.metadata.get("work_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        pr_number = self.metadata.get("pr_number")
        if pr_number is not None:
            try:
                return f"pr:{int(pr_number)}"
            except (TypeError, ValueError):
                pass
        if _looks_like_work_id(self.task_id):
            return self.task_id
        return None

    def owner_report(self) -> str:
        return (
            f"branch '{self.branch}' is leased by {self.owner_agent}/"
            f"{self.owner_session_id} (lease {self.lease_id}, task {self.task_id}, "
            f"expires {self.expires_at})"
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _looks_like_work_id(value: str | None) -> bool:
    if not value:
        return False
    prefix, sep, suffix = value.partition(":")
    return bool(sep and suffix and prefix in {"pr", "issue", "factory", "branch"})


def _resolve_work_id(args: argparse.Namespace, branch: str) -> str | None:
    explicit = getattr(args, "work_id", None)
    if explicit and explicit.strip():
        return explicit.strip()
    if getattr(args, "pr", None) is not None:
        return f"pr:{args.pr}"
    if getattr(args, "task_id", None) and _looks_like_work_id(args.task_id):
        return args.task_id
    return None


def _lease_matches_work_id(lease: LeaseRow, work_id: str | None) -> bool:
    if work_id is None:
        return True
    if work_id.startswith("branch:"):
        _prefix, _sep, branch_name = work_id.partition(":")
        return lease.branch == branch_name
    return lease.work_id == work_id or lease.task_id == work_id


def _lease_allows_branch_owner_fallback(lease: LeaseRow) -> bool:
    work_id = lease.work_id
    return work_id is None or work_id.startswith("branch:")


def _advisory_exit_code(args: argparse.Namespace, ok: bool) -> int:
    return 0 if ok or getattr(args, "advisory", False) else 1


def _sidecar_matches(record: dict[str, Any] | None, *, branch: str, work_id: str | None) -> bool:
    if not record:
        return False
    sidecar_branch = str(record.get("branch") or "").strip()
    if sidecar_branch and sidecar_branch != branch:
        return False
    sidecar_work_id = str(record.get("work_id") or "").strip()
    return not (work_id and sidecar_work_id and sidecar_work_id != work_id)


def resolve_db_path(repo_root: Path, explicit: str | None = None) -> Path:
    """Mirror ``DevCoordinationStore``'s DB resolution without importing it."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_db = os.environ.get(DB_ENV_VAR, "").strip()
    if env_db:
        configured = Path(env_db).expanduser()
        return configured if configured.is_absolute() else (repo_root / configured).resolve()
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        raise StoreUnreachableError(
            proc.stderr.strip() or f"failed to resolve git common dir for {repo_root}"
        )
    common_dir = Path(proc.stdout.strip()).resolve()
    return common_dir / "aragora-agent-state" / "dev_coordination.db"


def resolve_session_id(explicit: str | None = None) -> tuple[str, bool]:
    """Return (session_id, is_stable). Falls back to user@host with a warning."""
    if explicit and explicit.strip():
        return explicit.strip(), True
    for env_name in SESSION_ENV_VARS:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value, True
    agent = ""
    for env_name in AGENT_ENV_VARS:
        value = os.environ.get(env_name, "").strip()
        if value:
            agent = value
            break
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - getpass env-dependent
        user = "unknown"
    host = socket.gethostname().split(".")[0]
    prefix = agent or "session"
    return f"{prefix}-{user}@{host}", False


def resolve_branch(repo_root: Path, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    branch = proc.stdout.strip()
    if proc.returncode != 0 or not branch or branch == "HEAD":
        raise StoreUnreachableError(f"could not determine current branch in {repo_root}")
    return branch


def _query_lease_rows(db_path: Path, branch: str, *, immutable: bool) -> list[sqlite3.Row]:
    uri = f"file:{db_path}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT lease_id, task_id, title, owner_agent, owner_session_id, branch,"
            " worktree_path, status, created_at, expires_at, metadata_json"
            " FROM leases WHERE branch = ? AND status = 'active'"
            " ORDER BY created_at ASC, lease_id ASC",
            (branch,),
        ).fetchall()
    finally:
        conn.close()


def leases_for_branch(db_path: Path, branch: str) -> list[LeaseRow]:
    """Read-only query of active leases for ``branch``.

    A missing DB file means the store has never been initialised, which we
    treat as "no leases" rather than unreachable (deterministic for fresh
    checkouts).

    Opening a WAL-mode DB with ``mode=ro`` still needs the ``-wal``/``-shm``
    sidecars: sqlite must create them when absent (a clean close removes
    them; a copied DB lacks them), which fails when the store directory is
    not writable by the invoking UID. On ``OperationalError`` we therefore
    retry with ``immutable=1``, which skips WAL entirely — acceptable
    staleness for an advisory pre-check, since claims go through the store
    anyway. Only if both attempts fail is the store treated as unreachable.
    """
    if not db_path.exists():
        return []
    try:
        rows = _query_lease_rows(db_path, branch, immutable=False)
    except sqlite3.OperationalError:
        try:
            rows = _query_lease_rows(db_path, branch, immutable=True)
        except sqlite3.Error as exc:
            raise StoreUnreachableError(f"cannot read lease store {db_path}: {exc}") from exc
    except sqlite3.Error as exc:
        raise StoreUnreachableError(f"cannot read lease store {db_path}: {exc}") from exc
    leases = [
        LeaseRow(
            lease_id=row["lease_id"],
            task_id=row["task_id"],
            title=row["title"],
            owner_agent=row["owner_agent"],
            owner_session_id=row["owner_session_id"],
            branch=row["branch"],
            worktree_path=row["worktree_path"],
            status=row["status"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            metadata=_json_obj(row["metadata_json"]),
        )
        for row in rows
    ]
    return leases


def active_leases_for_branch(db_path: Path, branch: str) -> list[LeaseRow]:
    """Read-only query of active, unexpired leases for ``branch``."""
    leases = leases_for_branch(db_path, branch)
    return [lease for lease in leases if not lease.is_expired]


def _load_store(repo_root: Path, db_path: Path) -> Any:
    """Import the live store lazily (mutation paths only — this is slow)."""
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from aragora.nomic.dev_coordination import DevCoordinationStore
    except Exception as exc:
        raise StoreUnreachableError(f"cannot import dev_coordination store: {exc}") from exc
    try:
        return DevCoordinationStore(repo_root=repo_root, db_path=db_path)
    except Exception as exc:
        raise StoreUnreachableError(f"cannot open dev_coordination store: {exc}") from exc


def record_lane_lease(
    repo_root: Path,
    lane_id: str,
    *,
    branch: str,
    lease_id: str | None,
    session_id: str,
    work_id: str | None = None,
) -> Path:
    """Record (or clear, when ``lease_id`` is None) a lane→lease mapping.

    Non-invasive agent-bridge adapter: instead of changing ``LaneRecord``,
    the mapping lives in a sidecar JSON file keyed by lane id.
    """
    sidecar = repo_root / LANE_LEASES_RELPATH
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    lock_path = sidecar.with_name(f"{sidecar.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        payload: dict[str, Any] = {}
        if sidecar.exists():
            try:
                loaded = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = {}
        if lease_id is None:
            payload.pop(lane_id, None)
        else:
            payload[lane_id] = {
                "branch": branch,
                "lease_id": lease_id,
                "owner_session_id": session_id,
                "updated_at": _utcnow().isoformat(),
            }
            if work_id:
                payload[lane_id]["work_id"] = work_id
        fd, tmp_name = tempfile.mkstemp(dir=str(sidecar.parent), prefix=".lane-leases-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, sidecar)
        except OSError:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return sidecar


def read_lane_lease(repo_root: Path, lane_id: str) -> dict[str, Any] | None:
    sidecar = repo_root / LANE_LEASES_RELPATH
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    record = payload.get(lane_id)
    return record if isinstance(record, dict) else None


def _emit(
    args: argparse.Namespace,
    *,
    ok: bool,
    action: str,
    detail: str,
    reason: str | None = None,
    work_id: str | None = None,
    branch: str | None = None,
    lease_id: str | None = None,
    owner_session_id: str | None = None,
    **extra: Any,
) -> None:
    if getattr(args, "json", False):
        payload = {
            "ok": ok,
            "action": action,
            "reason": reason,
            "work_id": work_id,
            "branch": branch,
            "lease_id": lease_id,
            "owner_session_id": owner_session_id,
            "detail": detail,
            **extra,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))  # noqa: T201
    else:
        prefix = "OK" if ok else "BLOCKED"
        print(f"{prefix} [{action}] {detail}")  # noqa: T201


def _warn_fail_open(args: argparse.Namespace, reason: str) -> int:
    if args.strict:
        _emit(
            args,
            ok=False,
            action="store-unreachable",
            reason=REASON_STORE_UNREACHABLE,
            detail=f"{reason} (--strict: failing closed)",
        )
        return _advisory_exit_code(args, False)
    message = f"WARNING: lease store unreachable ({reason}) — proceeding fail-open (v0; use --strict to fail closed)"
    print(message, file=sys.stderr)  # noqa: T201
    if args.json:
        _emit(
            args,
            ok=True,
            action="store-unreachable",
            reason=REASON_STORE_UNREACHABLE,
            detail=message,
        )
    return 0


def _conflict_line(conflict: dict[str, Any], branch: str) -> str:
    """One-line owner report from a ``LeaseConflictError`` conflict dict."""
    if conflict.get("source") == "fleet_claim":
        return (
            f"branch '{branch}' blocked by fleet claim on {conflict.get('path', '?')} "
            f"(session {conflict.get('session_id', '?')})"
        )
    return (
        f"branch '{conflict.get('branch') or branch}' is leased by "
        f"{conflict.get('owner_agent', '?')}/{conflict.get('owner_session_id', '?')} "
        f"(lease {conflict.get('lease_id', '?')}, task {conflict.get('task_id', '?')}, "
        f"expires {conflict.get('expires_at', '?')})"
    )


def _claim(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    db_path: Path,
    branch: str,
    session_id: str,
) -> tuple[int, str | None]:
    """Claim the branch lease via the live store. Returns (exit_code, lease_id).

    Goes straight to ``store.claim_lease`` (no read-only pre-check) so the
    store gets to reap expired and dead-worker leases first — a stale lease
    with a future ``expires_at`` must not squat the branch until TTL.
    """
    store = _load_store(repo_root, db_path)
    metadata: dict[str, Any] = {"claimed_via": "check_work_lease"}
    if args.pr is not None:
        metadata["pr_number"] = args.pr
    work_id = _resolve_work_id(args, branch)
    if work_id:
        metadata["work_id"] = work_id
    from aragora.nomic.dev_coordination import LeaseConflictError

    branch_lock = BRANCH_LOCK_GLOB_TEMPLATE.format(branch=branch)
    try:
        lease = store.claim_lease(
            task_id=args.task_id or work_id or f"branch:{branch}",
            title=args.title or f"Push lease for {branch}",
            owner_agent=args.agent,
            owner_session_id=session_id,
            branch=branch,
            worktree_path=str(args.worktree or repo_root),
            allowed_globs=[branch_lock, *args.write_scope],
            claimed_paths=list(args.path),
            ttl_hours=args.ttl_hours,
            metadata=metadata,
        )
    except LeaseConflictError as exc:
        _emit(
            args,
            ok=False,
            action="claim",
            reason=REASON_WRONG_OWNER,
            work_id=work_id,
            branch=branch,
            detail=f"LEASE CONFLICT: {_conflict_line(exc.conflicts[0], branch)}",
            conflicts=exc.conflicts,
        )
        return _advisory_exit_code(args, False), None
    # Post-claim double-check: helper-vs-helper conflicts are caught
    # transactionally by the branch-lock glob above, but a store-direct
    # claimant (e.g. swarm) holding this branch with a non-overlapping file
    # scope is not. If a surviving foreign lease for this branch predates
    # ours (earlier created_at wins), back off and release ours.
    contenders = active_leases_for_branch(store.db_path, branch)
    for contender in contenders:
        if contender.lease_id == lease.lease_id:
            break
        if contender.owner_session_id != session_id:
            store.release_lease(lease.lease_id)
            _emit(
                args,
                ok=False,
                action="claim",
                reason=REASON_WRONG_OWNER,
                work_id=work_id,
                branch=branch,
                lease_id=contender.lease_id,
                owner_session_id=contender.owner_session_id,
                detail=f"LEASE CONFLICT: {contender.owner_report()}",
            )
            return _advisory_exit_code(args, False), None
    _emit(
        args,
        ok=True,
        action="claim",
        work_id=work_id,
        branch=branch,
        detail=f"claimed lease {lease.lease_id} for branch '{branch}' (session {session_id})",
        lease_id=lease.lease_id,
        owner_session_id=session_id,
    )
    return 0, lease.lease_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lease-rule preflight: verify this session holds the work lease for a branch "
        "in the live dev_coordination store before pushing (#8851 item 2)."
    )
    parser.add_argument("branch", nargs="?", default=None, help="Branch (default: current branch)")
    parser.add_argument(
        "--pr", type=int, default=None, help="Related PR number (recorded in lease metadata)"
    )
    parser.add_argument("--repo", default=".", help="Repository root (default: cwd)")
    parser.add_argument("--db", default=None, help="Explicit SQLite path (default: live store)")
    parser.add_argument(
        "--session-id", default=None, help="Session identity (default: $ARAGORA_SESSION_ID)"
    )
    parser.add_argument(
        "--agent", default=None, help="Agent name for claims (default: $ARAGORA_AGENT or 'unknown')"
    )
    parser.add_argument("--claim", action="store_true", help="Claim the lease if nobody holds it")
    parser.add_argument("--renew", action="store_true", help="Renew (heartbeat) the held lease")
    parser.add_argument("--release", action="store_true", help="Release the held lease")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing matching lease; never write lease or lane state",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report the same check result but exit 0 on failures",
    )
    parser.add_argument(
        "--work-id",
        default=None,
        help="Stable work identity: pr:<n>, issue:<n>, factory:<id>, or branch:<name>",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Fail closed when the store is unreachable"
    )
    parser.add_argument(
        "--task-id", default=None, help="Task id for claims (default: branch:<branch>)"
    )
    parser.add_argument("--title", default=None, help="Lease title for claims")
    parser.add_argument(
        "--worktree", default=None, help="Worktree path for claims (default: repo root)"
    )
    parser.add_argument(
        "--ttl-hours", type=float, default=DEFAULT_TTL_HOURS, help="Lease TTL for claims/renewals"
    )
    parser.add_argument(
        "--path", action="append", default=[], help="Claimed path (repeatable, for scoped claims)"
    )
    parser.add_argument(
        "--write-scope", action="append", default=[], help="Allowed glob (repeatable)"
    )
    parser.add_argument(
        "--record-lane",
        default=None,
        metavar="LANE_ID",
        help="Record the lease in the agent-bridge lane-leases sidecar",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)

    if args.release and (args.claim or args.renew):
        parser.error("--release cannot be combined with --claim/--renew")
    if args.verify_only and (args.claim or args.renew or args.release):
        parser.error("--verify-only cannot be combined with --claim/--renew/--release")
    if args.advisory and (args.claim or args.renew or args.release):
        parser.error("--advisory cannot be combined with --claim/--renew/--release")

    repo_root = Path(args.repo).expanduser().resolve()
    session_id, stable = resolve_session_id(args.session_id)
    if not stable:
        print(
            f"WARNING: no ARAGORA_SESSION_ID set — using host-level identity '{session_id}'. "
            "Set ARAGORA_SESSION_ID for per-session ownership.",
            file=sys.stderr,
        )  # noqa: T201
    if not args.agent:
        args.agent = next(
            (
                os.environ.get(name, "").strip()
                for name in AGENT_ENV_VARS
                if os.environ.get(name, "").strip()
            ),
            "unknown",
        )

    try:
        branch = resolve_branch(repo_root, args.branch)
        db_path = resolve_db_path(repo_root, args.db)
        all_leases = leases_for_branch(db_path, branch)
    except (StoreUnreachableError, subprocess.TimeoutExpired, OSError) as exc:
        return _warn_fail_open(args, str(exc))

    work_id = _resolve_work_id(args, branch)
    leases = [lease for lease in all_leases if not lease.is_expired]
    matching_leases = [lease for lease in leases if _lease_matches_work_id(lease, work_id)]
    expired_matching_leases = [
        lease for lease in all_leases if lease.is_expired and _lease_matches_work_id(lease, work_id)
    ]
    mine = [lease for lease in matching_leases if lease.owner_session_id == session_id]
    branch_mine = [lease for lease in leases if lease.owner_session_id == session_id]
    branch_theirs = [lease for lease in leases if lease.owner_session_id != session_id]
    if work_id and not mine and not matching_leases:
        mine = [lease for lease in branch_mine if _lease_allows_branch_owner_fallback(lease)]

    if args.verify_only:
        sidecar = read_lane_lease(repo_root, args.record_lane) if args.record_lane else None
        distinct_owners = {lease.owner_session_id for lease in matching_leases}
        if len(distinct_owners) > 1:
            first = matching_leases[0]
            _emit(
                args,
                ok=False,
                action="check",
                reason=REASON_AMBIGUOUS_OWNER,
                work_id=work_id,
                branch=branch,
                lease_id=first.lease_id,
                owner_session_id=first.owner_session_id,
                detail=f"ambiguous active leases for branch '{branch}' and work_id '{work_id}'",
                owners=sorted(distinct_owners),
            )
            return _advisory_exit_code(args, False)
        if branch_theirs:
            _emit(
                args,
                ok=False,
                action="check",
                reason=REASON_WRONG_OWNER,
                work_id=work_id,
                branch=branch,
                lease_id=branch_theirs[0].lease_id,
                owner_session_id=branch_theirs[0].owner_session_id,
                detail=f"LEASE CONFLICT: {branch_theirs[0].owner_report()}",
                owner=dataclasses.asdict(branch_theirs[0]),
            )
            return _advisory_exit_code(args, False)
        if mine:
            _emit(
                args,
                ok=True,
                action="check",
                work_id=work_id,
                branch=branch,
                lease_id=mine[0].lease_id,
                owner_session_id=session_id,
                detail=f"holding lease {mine[0].lease_id} for branch '{branch}' (session {session_id})",
            )
            return 0
        if expired_matching_leases:
            first = expired_matching_leases[0]
            _emit(
                args,
                ok=False,
                action="check",
                reason=REASON_EXPIRED_LEASE,
                work_id=work_id,
                branch=branch,
                lease_id=first.lease_id,
                owner_session_id=first.owner_session_id,
                detail=(
                    f"lease {first.lease_id} for branch '{branch}' expired at {first.expires_at}"
                ),
            )
            return _advisory_exit_code(args, False)
        if _sidecar_matches(sidecar, branch=branch, work_id=work_id):
            _emit(
                args,
                ok=False,
                action="check",
                reason=REASON_BRIDGE_ONLY_NO_DEV_LEASE,
                work_id=work_id,
                branch=branch,
                lease_id=str(sidecar.get("lease_id") or "") or None,
                owner_session_id=str(sidecar.get("owner_session_id") or "") or None,
                detail=(
                    f"lane '{args.record_lane}' has bridge lease metadata but no matching "
                    "dev_coordination lease"
                ),
            )
            return _advisory_exit_code(args, False)
        _emit(
            args,
            ok=False,
            action="check",
            reason=REASON_MISSING_LEASE,
            work_id=work_id,
            branch=branch,
            detail=(
                f"no active lease held for branch '{branch}'"
                + (f" and work_id '{work_id}'" if work_id else "")
            ),
        )
        return _advisory_exit_code(args, False)

    # Foreign leases block immediately on the read-only paths. The --claim
    # path (when we hold nothing) must NOT short-circuit here: it goes to
    # store.claim_lease first so the store can reap expired and dead-worker
    # leases instead of letting them squat the branch until TTL.
    if branch_theirs and not (args.claim and not branch_mine):
        _emit(
            args,
            ok=False,
            action="check",
            reason=REASON_WRONG_OWNER,
            work_id=work_id,
            branch=branch,
            lease_id=branch_theirs[0].lease_id,
            owner_session_id=branch_theirs[0].owner_session_id,
            detail=f"LEASE CONFLICT: {branch_theirs[0].owner_report()}",
            owner=dataclasses.asdict(branch_theirs[0]),
        )
        return _advisory_exit_code(args, False)

    try:
        if args.release:
            if not mine:
                _emit(
                    args,
                    ok=True,
                    action="release",
                    work_id=work_id,
                    branch=branch,
                    detail=f"no active lease held for branch '{branch}' (no-op)",
                )
            else:
                store = _load_store(repo_root, db_path)
                for lease in mine:
                    store.release_lease(lease.lease_id)
                _emit(
                    args,
                    ok=True,
                    action="release",
                    work_id=work_id,
                    branch=branch,
                    detail=f"released lease {mine[0].lease_id} for branch '{branch}'",
                    lease_id=mine[0].lease_id,
                    owner_session_id=session_id,
                )
            if args.record_lane:
                record_lane_lease(
                    repo_root, args.record_lane, branch=branch, lease_id=None, session_id=session_id
                )
            return 0

        lease_id: str | None = mine[0].lease_id if mine else None
        if mine and args.renew:
            store = _load_store(repo_root, db_path)
            renewed = store.heartbeat_lease(mine[0].lease_id, args.ttl_hours)
            _emit(
                args,
                ok=True,
                action="renew",
                work_id=work_id,
                branch=branch,
                detail=f"renewed lease {renewed.lease_id} for branch '{branch}' until {renewed.expires_at}",
                lease_id=renewed.lease_id,
                owner_session_id=session_id,
            )
        elif mine:
            _emit(
                args,
                ok=True,
                action="check",
                work_id=work_id,
                branch=branch,
                detail=f"holding lease {mine[0].lease_id} for branch '{branch}' (session {session_id})",
                lease_id=mine[0].lease_id,
                owner_session_id=session_id,
            )
        elif args.claim:
            code, lease_id = _claim(
                args, repo_root=repo_root, db_path=db_path, branch=branch, session_id=session_id
            )
            if code != 0:
                return code
        elif args.renew:
            _emit(
                args,
                ok=False,
                action="renew",
                reason=REASON_MISSING_LEASE,
                work_id=work_id,
                branch=branch,
                detail=f"no active lease held for branch '{branch}' — claim one with --claim",
            )
            return _advisory_exit_code(args, False)
        else:
            _emit(
                args,
                ok=False,
                action="check",
                reason=REASON_MISSING_LEASE,
                work_id=work_id,
                branch=branch,
                detail=f"no active lease held for branch '{branch}' (session {session_id}) — run with --claim before pushing",
            )
            return _advisory_exit_code(args, False)

        if args.record_lane and lease_id:
            record_lane_lease(
                repo_root,
                args.record_lane,
                branch=branch,
                lease_id=lease_id,
                session_id=session_id,
                work_id=work_id,
            )
        return 0
    except (StoreUnreachableError, subprocess.TimeoutExpired, OSError) as exc:
        return _warn_fail_open(args, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
