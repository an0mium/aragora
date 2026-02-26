"""Fleet coordination monitor and merge-queue enforcement for worktree orchestration."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any
from uuid import uuid4

from aragora.nomic.session_manifest import SessionManifest
from aragora.worktree.lifecycle import WorktreeLifecycleService

logger = logging.getLogger(__name__)

UTC = timezone.utc
DEFAULT_TAIL_LINES = 200
MAX_TAIL_LINES = 5000
STATE_DIR_NAME = ".aragora_coordination"
CLAIMS_FILE_NAME = "claims.json"
MERGE_QUEUE_FILE_NAME = "merge_queue.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_path(path: str) -> str:
    normalized = PurePosixPath(path.strip()).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


class FleetCoordinator:
    """Collects coordination state and enforces claims/queue invariants."""

    def __init__(self, repo_root: Path | None = None):
        root = (repo_root or Path.cwd()).resolve()
        self.repo_root = self._resolve_repo_root(root)
        self._lifecycle = WorktreeLifecycleService(repo_root=self.repo_root)
        self._manifest = SessionManifest(repo_root=self.repo_root)
        self._state_dir = self.repo_root / STATE_DIR_NAME
        self._claims_file = self._state_dir / CLAIMS_FILE_NAME
        self._merge_queue_file = self._state_dir / MERGE_QUEUE_FILE_NAME

    def _resolve_repo_root(self, path: Path) -> Path:
        proc = subprocess.run(  # noqa: S603 -- fixed command, no shell
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).resolve()
        return path

    def _run_cmd(self, cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 -- fixed command, no shell
            cmd,
            cwd=cwd or self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def _run_git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return self._run_cmd(["git", *args], cwd=cwd)

    def _run_json_cmd(self, cmd: list[str], cwd: Path | None = None) -> Any | None:
        try:
            proc = self._run_cmd(cmd, cwd=cwd)
        except FileNotFoundError:
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None

    def _git_worktree_map(self) -> dict[str, dict[str, Any]]:
        proc = self._run_git("worktree", "list", "--porcelain")
        if proc.returncode != 0:
            return {}

        out: dict[str, dict[str, Any]] = {}
        current_path: str | None = None
        current_branch: str | None = None
        detached = False

        def flush() -> None:
            nonlocal current_path, current_branch, detached
            if current_path:
                out[current_path] = {
                    "branch": current_branch,
                    "detached": detached,
                }
            current_path = None
            current_branch = None
            detached = False

        for raw in proc.stdout.splitlines():
            line = raw.strip()
            if not line:
                flush()
                continue
            if line.startswith("worktree "):
                flush()
                current_path = str(Path(line[len("worktree ") :]).resolve())
                continue
            if line.startswith("branch refs/heads/"):
                current_branch = line[len("branch refs/heads/") :]
                continue
            if line == "detached":
                detached = True

        flush()
        return out

    def _load_json_file(self, path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(fallback)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(fallback)
        if not isinstance(data, dict):
            return dict(fallback)
        merged = dict(fallback)
        merged.update(data)
        return merged

    def _save_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = _utc_now_iso()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load_claim_store(self) -> dict[str, Any]:
        data = self._load_json_file(
            self._claims_file,
            {"version": 1, "updated_at": "", "claims": []},
        )
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        data["claims"] = claims
        return data

    def _save_claim_store(self, data: dict[str, Any]) -> None:
        self._save_json_file(self._claims_file, data)

    def _load_merge_queue_store(self) -> dict[str, Any]:
        data = self._load_json_file(
            self._merge_queue_file,
            {"version": 1, "updated_at": "", "items": []},
        )
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        data["items"] = items
        return data

    def _save_merge_queue_store(self, data: dict[str, Any]) -> None:
        self._save_json_file(self._merge_queue_file, data)

    def _normalize_claim(self, claim_path: str) -> str:
        p = Path(claim_path)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(self.repo_root.resolve())
                return rel.as_posix()
            except ValueError:
                return p.as_posix()
        return _normalize_path(claim_path)

    def _manifest_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            sessions = self._manifest.list_active()
        except (OSError, ValueError, TypeError):
            sessions = []

        for s in sessions:
            worktree = str(Path(s.worktree).resolve()) if s.worktree else ""
            entries.append(
                {
                    "track": s.track,
                    "worktree": worktree,
                    "agent": s.agent,
                    "goal": s.current_goal,
                    "files_claimed": list(s.files_claimed),
                    "status": s.status,
                    "pid": s.pid,
                    "started_at": s.started_at,
                }
            )
        return entries

    def _autopilot_entries(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        managed_dirs = self._lifecycle.discover_managed_dirs(None)
        for managed_dir in managed_dirs:
            state_file = (self.repo_root / managed_dir / "state.json").resolve()
            data = self._load_json_file(state_file, {"sessions": []})
            for session in data.get("sessions", []):
                if not isinstance(session, dict):
                    continue
                path = str(Path(str(session.get("path", ""))).resolve())
                sessions.append(
                    {
                        "session_id": str(session.get("session_id", "")).strip(),
                        "agent": str(session.get("agent", "")).strip() or "codex",
                        "branch": str(session.get("branch", "")).strip(),
                        "path": path,
                        "created_at": session.get("created_at", ""),
                        "last_seen_at": session.get("last_seen_at", ""),
                        "reconcile_status": session.get("reconcile_status"),
                    }
                )
        return sessions

    def _git_head_sha(self, worktree_path: Path) -> str | None:
        proc = self._run_git("rev-parse", "HEAD", cwd=worktree_path)
        if proc.returncode != 0:
            return None
        sha = proc.stdout.strip()
        return sha or None

    def _git_remote_sha(self, branch: str) -> str | None:
        proc = self._run_git("rev-parse", f"origin/{branch}")
        if proc.returncode != 0:
            return None
        sha = proc.stdout.strip()
        return sha or None

    @staticmethod
    def _normalize_check_state(value: str) -> str:
        state = value.strip().lower()
        if state in {"success", "pass", "passed", "neutral", "skipping", "skipped", "complete"}:
            return "success"
        if state in {"failure", "failed", "error", "cancelled", "timed_out", "action_required"}:
            return "failure"
        if state in {"pending", "queued", "in_progress", "waiting", "requested"}:
            return "pending"
        return "unknown"

    def _resolve_pr_status(self, branch: str | None) -> dict[str, Any]:
        default = {
            "pr": None,
            "required_checks_state": "unknown",
            "required_checks": [],
            "failing_checks": [],
        }
        if not branch:
            return {**default, "required_checks_state": "none"}

        prs = self._run_json_cmd(
            [
                "gh",
                "pr",
                "list",
                "--head",
                branch,
                "--state",
                "open",
                "--json",
                "number,url,title,isDraft,headRefName,baseRefName",
                "--limit",
                "1",
            ]
        )
        if not isinstance(prs, list) or not prs:
            return default

        pr = prs[0]
        number = pr.get("number")
        checks_payload = None
        if number:
            checks_payload = self._run_json_cmd(
                [
                    "gh",
                    "pr",
                    "checks",
                    str(number),
                    "--required",
                    "--json",
                    "name,state,link,workflow",
                ]
            )

        checks: list[dict[str, Any]] = []
        failing: list[dict[str, Any]] = []
        states: list[str] = []
        if isinstance(checks_payload, list):
            for row in checks_payload:
                if not isinstance(row, dict):
                    continue
                normalized = self._normalize_check_state(str(row.get("state", "")))
                check_row = {
                    "name": row.get("name", ""),
                    "state": row.get("state", ""),
                    "normalized_state": normalized,
                    "link": row.get("link"),
                    "workflow": row.get("workflow"),
                }
                checks.append(check_row)
                states.append(normalized)
                if normalized == "failure":
                    failing.append(check_row)

        if failing:
            state = "red"
        elif states and all(s == "success" for s in states):
            state = "green"
        elif any(s == "pending" for s in states):
            state = "pending"
        elif states:
            state = "unknown"
        else:
            state = "unknown"

        return {
            "pr": {
                "number": pr.get("number"),
                "url": pr.get("url"),
                "title": pr.get("title"),
                "is_draft": pr.get("isDraft", False),
                "head_ref": pr.get("headRefName"),
                "base_ref": pr.get("baseRefName"),
            },
            "required_checks_state": state,
            "required_checks": checks,
            "failing_checks": failing,
        }

    @staticmethod
    def _tail_file_lines(path: Path, tail_lines: int) -> list[str]:
        if tail_lines <= 0:
            return []
        tail: deque[str] = deque(maxlen=tail_lines)
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    tail.append(line.rstrip("\n"))
        except OSError:
            return []
        return list(tail)

    def _discover_log_file(self, worktree_path: Path, session: dict[str, Any]) -> Path | None:
        candidates: list[Path] = []

        explicit = session.get("log_path")
        if isinstance(explicit, str) and explicit.strip():
            candidates.append(Path(explicit).expanduser().resolve())

        candidates.extend(
            [
                worktree_path / ".sprint-agent.log",
                worktree_path / ".codex.log",
                worktree_path / ".aragora-agent.log",
                worktree_path / "logs" / "agent.log",
                worktree_path / "logs" / "session.log",
            ]
        )

        existing = [p for p in candidates if p.exists() and p.is_file()]
        if not existing:
            return None
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing[0]

    def _manifest_claim_map(self) -> dict[str, set[str]]:
        claims: dict[str, set[str]] = defaultdict(set)
        for entry in self._manifest_entries():
            track = str(entry.get("track", "")).strip()
            if not track:
                continue
            owner = f"track:{track}"
            for path in entry.get("files_claimed", []):
                if not isinstance(path, str) or not path.strip():
                    continue
                claims[owner].add(self._normalize_claim(path))
        return claims

    def _stored_claim_map(self) -> dict[str, set[str]]:
        claims: dict[str, set[str]] = defaultdict(set)
        data = self._load_claim_store()
        for row in data.get("claims", []):
            if not isinstance(row, dict):
                continue
            owner = str(row.get("owner", "")).strip()
            if not owner:
                continue
            for path in row.get("paths", []):
                if isinstance(path, str) and path.strip():
                    claims[owner].add(self._normalize_claim(path))
        return claims

    def _effective_claim_map(self) -> dict[str, set[str]]:
        combined: dict[str, set[str]] = defaultdict(set)
        for owner, paths in self._manifest_claim_map().items():
            combined[owner].update(paths)
        for owner, paths in self._stored_claim_map().items():
            combined[owner].update(paths)
        return combined

    @staticmethod
    def _claim_conflicts_from_map(claim_map: dict[str, set[str]]) -> list[dict[str, Any]]:
        by_path: dict[str, list[str]] = defaultdict(list)
        for owner, paths in claim_map.items():
            for path in sorted(paths):
                by_path[path].append(owner)

        conflicts: list[dict[str, Any]] = []
        for path, owners in sorted(by_path.items()):
            unique = sorted(set(owners))
            if len(unique) < 2:
                continue
            conflicts.append(
                {
                    "path": path,
                    "owners": unique,
                }
            )
        return conflicts

    def get_claims(self) -> dict[str, Any]:
        claim_map = self._effective_claim_map()
        claims_rows = [
            {
                "owner": owner,
                "paths": sorted(paths),
            }
            for owner, paths in sorted(claim_map.items())
        ]
        conflicts = self._claim_conflicts_from_map(claim_map)
        return {
            "generated_at": _utc_now_iso(),
            "claims": claims_rows,
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        }

    def claim_paths(
        self, owner: str, paths: list[str], *, override: bool = False
    ) -> dict[str, Any]:
        owner_key = owner.strip()
        if not owner_key:
            return {
                "ok": False,
                "applied": False,
                "error": "owner is required",
            }

        normalized_paths = sorted({self._normalize_claim(path) for path in paths if path.strip()})
        if not normalized_paths:
            return {
                "ok": False,
                "applied": False,
                "error": "paths is required",
            }

        current = self._effective_claim_map()
        conflicts: list[dict[str, str]] = []
        for other_owner, other_paths in current.items():
            if other_owner == owner_key:
                continue
            overlap = sorted(set(normalized_paths) & other_paths)
            for path in overlap:
                conflicts.append({"path": path, "owner": other_owner})

        if conflicts and not override:
            return {
                "ok": False,
                "applied": False,
                "owner": owner_key,
                "paths": normalized_paths,
                "conflicts": conflicts,
                "message": "conflicting claims detected; set override=true to force",
            }

        store = self._load_claim_store()
        claims = [row for row in store.get("claims", []) if isinstance(row, dict)]
        existing: dict[str, Any] | None = None
        for row in claims:
            if str(row.get("owner", "")).strip() == owner_key:
                existing = row
                break

        if existing is None:
            existing = {"owner": owner_key, "paths": [], "updated_at": "", "override": False}
            claims.append(existing)

        updated = sorted(set(existing.get("paths", [])) | set(normalized_paths))
        existing["paths"] = updated
        existing["updated_at"] = _utc_now_iso()
        existing["override"] = bool(override)

        store["claims"] = claims
        self._save_claim_store(store)

        return {
            "ok": True,
            "applied": True,
            "owner": owner_key,
            "paths": updated,
            "conflicts": conflicts,
            "override": bool(override),
        }

    def _collect_sessions(
        self,
        *,
        tail_lines: int,
        include_logs: bool,
        include_checks: bool,
    ) -> list[dict[str, Any]]:
        worktrees = self._git_worktree_map()
        manifest_entries = self._manifest_entries()
        autopilot_entries = self._autopilot_entries()

        by_key: dict[str, dict[str, Any]] = {}

        for session in autopilot_entries:
            path = str(Path(str(session.get("path", ""))).resolve()) if session.get("path") else ""
            key = path or f"session:{session.get('session_id', uuid4().hex[:8])}"
            record = by_key.setdefault(
                key,
                {
                    "session_id": session.get("session_id") or "",
                    "owner": f"session:{session.get('session_id') or key}",
                    "track": "",
                    "agent": session.get("agent") or "codex",
                    "goal": "",
                    "pid": 0,
                    "worktree_path": path,
                    "branch": session.get("branch") or "",
                    "local_sha": None,
                    "remote_sha": None,
                    "drift": False,
                    "active": False,
                    "lock_file": False,
                    "started_at": session.get("created_at") or "",
                    "last_seen_at": session.get("last_seen_at") or "",
                    "claimed_paths": [],
                    "log_path": None,
                    "log_tail": [],
                    "pr": None,
                    "required_checks_state": "unknown",
                    "required_checks": [],
                    "failing_checks": [],
                },
            )
            if session.get("branch"):
                record["branch"] = session.get("branch")

        for session in manifest_entries:
            path = (
                str(Path(str(session.get("worktree", ""))).resolve())
                if session.get("worktree")
                else ""
            )
            key = path or f"track:{session.get('track', uuid4().hex[:6])}"
            record = by_key.setdefault(
                key,
                {
                    "session_id": session.get("track") or "",
                    "owner": f"track:{session.get('track')}",
                    "track": session.get("track") or "",
                    "agent": session.get("agent") or "codex",
                    "goal": session.get("goal") or "",
                    "pid": int(session.get("pid") or 0),
                    "worktree_path": path,
                    "branch": "",
                    "local_sha": None,
                    "remote_sha": None,
                    "drift": False,
                    "active": False,
                    "lock_file": False,
                    "started_at": session.get("started_at") or "",
                    "last_seen_at": "",
                    "claimed_paths": [],
                    "log_path": None,
                    "log_tail": [],
                    "pr": None,
                    "required_checks_state": "unknown",
                    "required_checks": [],
                    "failing_checks": [],
                },
            )
            track = str(session.get("track", "")).strip()
            if track:
                record["track"] = track
                record["owner"] = f"track:{track}"
                if not record.get("session_id"):
                    record["session_id"] = track
            if session.get("goal"):
                record["goal"] = session.get("goal")
            if session.get("agent"):
                record["agent"] = session.get("agent")
            if session.get("pid"):
                record["pid"] = int(session.get("pid") or 0)
            if not record.get("started_at") and session.get("started_at"):
                record["started_at"] = session.get("started_at")

        claims = self._effective_claim_map()
        branch_cache: dict[str, dict[str, Any]] = {}

        sessions_out: list[dict[str, Any]] = []
        for record in by_key.values():
            path_str = str(record.get("worktree_path", ""))
            path = Path(path_str) if path_str else None

            if not record.get("branch") and path_str in worktrees:
                branch = worktrees[path_str].get("branch")
                if branch:
                    record["branch"] = branch

            if path and path.exists():
                record["local_sha"] = self._git_head_sha(path)
                lock_file = path / ".codex_session_active"
                record["lock_file"] = lock_file.exists()
                record["active"] = bool(record["lock_file"] or path_str in worktrees)
            else:
                record["active"] = False
                record["lock_file"] = False

            branch = str(record.get("branch", "")).strip() or None
            if branch:
                record["remote_sha"] = self._git_remote_sha(branch)
            record["drift"] = bool(
                record.get("local_sha")
                and record.get("remote_sha")
                and record.get("local_sha") != record.get("remote_sha")
            )

            owner = str(record.get("owner", "")).strip()
            claimed = set(claims.get(owner, set()))
            track = str(record.get("track", "")).strip()
            if track:
                claimed.update(claims.get(f"track:{track}", set()))
            record["claimed_paths"] = sorted(claimed)

            if include_logs and tail_lines > 0 and path and path.exists():
                log_file = self._discover_log_file(path, record)
                if log_file:
                    record["log_path"] = str(log_file)
                    record["log_tail"] = self._tail_file_lines(log_file, tail_lines)

            if include_checks and branch:
                if branch not in branch_cache:
                    branch_cache[branch] = self._resolve_pr_status(branch)
                status = branch_cache[branch]
                record["pr"] = status.get("pr")
                record["required_checks_state"] = status.get("required_checks_state", "unknown")
                record["required_checks"] = status.get("required_checks", [])
                record["failing_checks"] = status.get("failing_checks", [])
            elif include_checks:
                record["required_checks_state"] = "none"

            sessions_out.append(record)

        sessions_out.sort(key=lambda row: (not bool(row.get("active")), str(row.get("owner", ""))))
        return sessions_out

    def fleet_logs(
        self, *, tail_lines: int = DEFAULT_TAIL_LINES, session_id: str | None = None
    ) -> dict[str, Any]:
        tail = max(1, min(tail_lines, MAX_TAIL_LINES))
        sessions = self._collect_sessions(tail_lines=tail, include_logs=True, include_checks=False)
        if session_id:
            sessions = [s for s in sessions if str(s.get("session_id")) == session_id]

        return {
            "generated_at": _utc_now_iso(),
            "tail_lines": tail,
            "sessions": [
                {
                    "session_id": row.get("session_id"),
                    "owner": row.get("owner"),
                    "worktree_path": row.get("worktree_path"),
                    "log_path": row.get("log_path"),
                    "log_tail": row.get("log_tail", []),
                }
                for row in sessions
            ],
        }

    def _queue_conflicts_for_owner(
        self,
        owner: str,
        claim_conflicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        for conflict in claim_conflicts:
            owners = conflict.get("owners", [])
            if owner in owners and len(owners) > 1:
                conflicts.append(conflict)
        return conflicts

    def get_merge_queue(self, *, sessions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        store = self._load_merge_queue_store()
        items = [row for row in store.get("items", []) if isinstance(row, dict)]

        if sessions is None:
            sessions = self._collect_sessions(tail_lines=0, include_logs=False, include_checks=True)

        by_owner = {str(row.get("owner", "")): row for row in sessions}
        by_session_id = {str(row.get("session_id", "")): row for row in sessions}
        claims = self.get_claims()
        claim_conflicts = claims.get("conflicts", [])

        annotated: list[dict[str, Any]] = []
        blocked = 0
        ready = 0

        for item in items:
            row = dict(item)
            owner = str(row.get("owner", "")).strip()
            sid = str(row.get("session_id", "")).strip()
            session = by_owner.get(owner) or by_session_id.get(sid)

            branch = str(row.get("branch", "")).strip()
            if not branch and session:
                branch = str(session.get("branch", "")).strip()
                row["branch"] = branch

            check_state = "unknown"
            if session:
                check_state = str(session.get("required_checks_state", "unknown"))
            elif branch:
                check_state = self._resolve_pr_status(branch).get(
                    "required_checks_state", "unknown"
                )

            blockers: list[str] = []
            owner_conflicts = self._queue_conflicts_for_owner(owner, claim_conflicts)
            if owner_conflicts and not row.get("override_claim_conflicts", False):
                blockers.append("claim_conflict")

            if check_state == "red":
                blockers.append("required_checks_red")

            row["live_required_checks_state"] = check_state
            row["claim_conflicts"] = owner_conflicts
            row["blockers"] = blockers
            row["can_advance"] = len(blockers) == 0

            if blockers:
                blocked += 1
            elif row.get("status") == "ready":
                ready += 1

            annotated.append(row)

        return {
            "generated_at": _utc_now_iso(),
            "items": annotated,
            "total": len(annotated),
            "blocked": blocked,
            "ready": ready,
        }

    def enqueue_merge(
        self,
        *,
        owner: str,
        branch: str,
        session_id: str | None = None,
        pr_number: int | None = None,
        override_claim_conflicts: bool = False,
    ) -> dict[str, Any]:
        owner_key = owner.strip()
        branch_name = branch.strip()
        if not owner_key:
            return {"ok": False, "error": "owner is required"}
        if not branch_name:
            return {"ok": False, "error": "branch is required"}

        store = self._load_merge_queue_store()
        items = [row for row in store.get("items", []) if isinstance(row, dict)]

        item = {
            "id": f"mq-{uuid4().hex[:10]}",
            "owner": owner_key,
            "session_id": session_id or "",
            "branch": branch_name,
            "pr_number": pr_number,
            "status": "queued",
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "override_claim_conflicts": bool(override_claim_conflicts),
            "last_error": None,
        }
        items.append(item)
        store["items"] = items
        self._save_merge_queue_store(store)

        return {
            "ok": True,
            "enqueued": True,
            "item": item,
        }

    def remove_merge_item(self, item_id: str) -> dict[str, Any]:
        item_key = item_id.strip()
        store = self._load_merge_queue_store()
        items = [row for row in store.get("items", []) if isinstance(row, dict)]
        kept = [row for row in items if str(row.get("id", "")) != item_key]
        removed = len(kept) != len(items)
        store["items"] = kept
        self._save_merge_queue_store(store)
        return {
            "ok": True,
            "removed": removed,
            "id": item_key,
        }

    def clear_merge_queue(self) -> dict[str, Any]:
        store = self._load_merge_queue_store()
        count = len(store.get("items", []))
        store["items"] = []
        self._save_merge_queue_store(store)
        return {
            "ok": True,
            "cleared": count,
        }

    def advance_merge_queue(self) -> dict[str, Any]:
        store = self._load_merge_queue_store()
        items = [row for row in store.get("items", []) if isinstance(row, dict)]
        if not items:
            return {
                "ok": True,
                "advanced": False,
                "reason": "queue_empty",
            }

        sessions = self._collect_sessions(tail_lines=0, include_logs=False, include_checks=True)
        queue = self.get_merge_queue(sessions=sessions)
        by_id = {str(row.get("id", "")): row for row in queue.get("items", [])}

        target_index = -1
        for idx, row in enumerate(items):
            if str(row.get("status", "queued")) in {"queued", "blocked"}:
                target_index = idx
                break

        if target_index < 0:
            return {
                "ok": True,
                "advanced": False,
                "reason": "no_pending_items",
            }

        target = items[target_index]
        item_id = str(target.get("id", ""))
        live = by_id.get(item_id, {})
        blockers = list(live.get("blockers", []))

        if blockers:
            target["status"] = "blocked"
            target["updated_at"] = _utc_now_iso()
            target["last_error"] = ",".join(blockers)
            store["items"] = items
            self._save_merge_queue_store(store)
            return {
                "ok": True,
                "advanced": False,
                "reason": "blocked",
                "item": target,
                "blockers": blockers,
            }

        target["status"] = "ready"
        target["advanced_at"] = _utc_now_iso()
        target["updated_at"] = _utc_now_iso()
        target["last_error"] = None
        store["items"] = items
        self._save_merge_queue_store(store)
        return {
            "ok": True,
            "advanced": True,
            "item": target,
        }

    def _actionable_failures(
        self,
        *,
        sessions: list[dict[str, Any]],
        claims: dict[str, Any],
        merge_queue: dict[str, Any],
    ) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []

        for conflict in claims.get("conflicts", []):
            failures.append(
                {
                    "priority": "P0",
                    "type": "claim_conflict",
                    "summary": f"Path '{conflict.get('path')}' claimed by multiple owners",
                    "owners": conflict.get("owners", []),
                }
            )

        for item in merge_queue.get("items", []):
            blockers = item.get("blockers", [])
            if "required_checks_red" in blockers:
                failures.append(
                    {
                        "priority": "P0",
                        "type": "merge_queue_blocked",
                        "summary": (
                            f"Merge queue item {item.get('id')} blocked: required checks red "
                            f"for branch {item.get('branch')}"
                        ),
                        "item_id": item.get("id"),
                    }
                )
            if "claim_conflict" in blockers:
                failures.append(
                    {
                        "priority": "P0",
                        "type": "merge_queue_claim_conflict",
                        "summary": (
                            f"Merge queue item {item.get('id')} blocked: claim conflicts "
                            f"for owner {item.get('owner')}"
                        ),
                        "item_id": item.get("id"),
                    }
                )

        for session in sessions:
            if session.get("required_checks_state") == "red":
                failures.append(
                    {
                        "priority": "P1",
                        "type": "required_checks_red",
                        "summary": (
                            f"Session {session.get('owner')} has failing required checks "
                            f"on branch {session.get('branch')}"
                        ),
                        "branch": session.get("branch"),
                    }
                )
            if session.get("drift"):
                failures.append(
                    {
                        "priority": "P2",
                        "type": "remote_drift",
                        "summary": (
                            f"Session {session.get('owner')} drifted: local SHA "
                            f"{session.get('local_sha')} != remote SHA {session.get('remote_sha')}"
                        ),
                        "branch": session.get("branch"),
                    }
                )

        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        failures.sort(
            key=lambda row: (priority_order.get(str(row.get("priority")), 9), row["summary"])
        )
        return failures

    def fleet_status(self, *, tail_lines: int = DEFAULT_TAIL_LINES) -> dict[str, Any]:
        tail = max(1, min(tail_lines, MAX_TAIL_LINES))
        sessions = self._collect_sessions(tail_lines=tail, include_logs=True, include_checks=True)
        claims = self.get_claims()
        merge_queue = self.get_merge_queue(sessions=sessions)
        failures = self._actionable_failures(
            sessions=sessions,
            claims=claims,
            merge_queue=merge_queue,
        )

        return {
            "generated_at": _utc_now_iso(),
            "tail_lines": tail,
            "total_sessions": len(sessions),
            "active_sessions": sum(1 for row in sessions if row.get("active")),
            "sessions": sessions,
            "claims": claims,
            "merge_queue": merge_queue,
            "actionable_failures": failures,
        }

    def write_report(
        self,
        *,
        tail_lines: int = DEFAULT_TAIL_LINES,
        output_dir: Path | None = None,
        status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = status or self.fleet_status(tail_lines=tail_lines)
        out_dir = (output_dir or (self.repo_root / "docs" / "status")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
        report_path = out_dir / f"fleet-status-{stamp}.md"

        lines: list[str] = []
        lines.append(f"# Fleet Status Report ({stamp})")
        lines.append("")
        lines.append(f"- Generated at: {payload.get('generated_at')}")
        lines.append(
            f"- Active sessions: {payload.get('active_sessions')}/{payload.get('total_sessions')}"
        )
        lines.append(f"- Claim conflicts: {payload.get('claims', {}).get('conflict_count', 0)}")
        lines.append(f"- Merge queue items: {payload.get('merge_queue', {}).get('total', 0)}")
        lines.append("")

        failures = payload.get("actionable_failures", [])
        lines.append("## Actionable Failures (Priority Order)")
        if failures:
            for item in failures:
                lines.append(f"- [{item.get('priority')}] {item.get('summary')}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("## Sessions")
        for row in payload.get("sessions", []):
            lines.append(
                f"- {row.get('owner')} | branch={row.get('branch')} | active={row.get('active')} "
                f"| checks={row.get('required_checks_state')} | drift={row.get('drift')}"
            )
        lines.append("")

        lines.append("## Merge Queue")
        for item in payload.get("merge_queue", {}).get("items", []):
            blockers = item.get("blockers", [])
            block_txt = ", ".join(blockers) if blockers else "none"
            lines.append(
                f"- {item.get('id')} | owner={item.get('owner')} | branch={item.get('branch')} "
                f"| status={item.get('status')} | blockers={block_txt}"
            )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {
            "ok": True,
            "report_path": str(report_path),
            "generated_at": payload.get("generated_at"),
            "actionable_failures": len(failures),
        }


def create_fleet_coordinator(repo_root: Path | None = None) -> FleetCoordinator:
    """Factory helper used by CLI and API handlers."""
    return FleetCoordinator(repo_root=repo_root)


__all__ = [
    "FleetCoordinator",
    "create_fleet_coordinator",
]
