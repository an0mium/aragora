#!/usr/bin/env python3
"""Read-only cross-agent overlap detector.

Consolidates the signals that tell an operator "who is working on what
right now?" into a single JSON or text payload so concurrent agents
(Factory Droid, Claude Code Desktop, Claude CLI, Codex CLI, Codex
Desktop cron automations, the aragora boss-loop, agent-bridge) do not
collide on the same branch, file, or worktree.

Inputs (each optional, never errors on missing sources):

- ``git worktree list --porcelain`` for managed worktrees + locked
  markers
- ``.aragora/dispatch_contracts/*.json`` for live dispatch contracts
- ``.aragora/issue_claims/*.json`` for claimed issues
- ``.aragora/work-leases/*.json`` for work board leases
- ``.aragora/fleet_coordination.json`` and ``fleet_coordination.lock``
- ``.aragora/automation-outbox/*.json`` for unpublished handoffs
- ``scripts/check_codex_desktop_automations.py --json`` for Codex
  Desktop cron status
- ``~/.codex/sessions/**/*.jsonl`` for Codex CLI session files
- ``scripts/agent_bridge.py processes --json --summary-only`` for live
  agent process census
- ``gh pr list --state open --json ...`` for open PRs (skippable via
  ``--skip-gh``)

The output's ``overlap_report`` block flags branches and worktree
paths that appear in more than one signal source so the operator can
quickly see whether a new session would overlap with an in-flight
agent.

This script is read-only:

- it makes no writes to any file
- it never deletes worktrees, branches, contracts, or leases
- it imports no aragora package (pure stdlib) so it works during
  partial bootstraps and on a freshly cloned checkout
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODEX_HOME = Path.home() / ".codex"
DEFAULT_MAX_AGE_MINUTES = 120
DEFAULT_MAX_PR_FETCH = 30
DEFAULT_GIT_TIMEOUT = 10
DEFAULT_SUBPROCESS_TIMEOUT = 20
DEFAULT_CODEX_SESSION_SCAN_LIMIT = 500
SCHEMA_VERSION = 1


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def _isoformat(ts: dt.datetime) -> str:
    return ts.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _file_age_minutes(path: Path, now: dt.datetime) -> float | None:
    try:
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
    except OSError:
        return None
    delta = now - mtime
    return max(0.0, delta.total_seconds() / 60.0)


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _stable_branch_name(value: object | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("refs/heads/"):
        text = text[len("refs/heads/") :]
    return text


def detect_worktrees(
    repo_root: Path, *, timeout: int = DEFAULT_GIT_TIMEOUT
) -> list[dict[str, Any]]:
    """Parse ``git worktree list --porcelain`` for the repo root."""
    if not (repo_root / ".git").exists() and not (repo_root / ".git").is_file():
        return []
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    out: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw in (proc.stdout or "").splitlines():
        line = raw.rstrip()
        if not line:
            if current:
                out.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree ") :]
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD ") :]
        elif line.startswith("branch "):
            current["branch"] = _stable_branch_name(line[len("branch ") :])
        elif line == "bare":
            current["bare"] = True
        elif line == "detached":
            current["detached"] = True
        elif line == "locked" or line.startswith("locked"):
            current["locked"] = True
    if current:
        out.append(current)
    return out


def detect_recent_jsonl_files(
    directory: Path,
    *,
    now: dt.datetime,
    max_age_minutes: float,
    limit: int = 50,
    suffix: str = ".json",
) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            (p for p in directory.iterdir() if p.is_file() and p.suffix == suffix),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    for path in candidates[:limit]:
        age = _file_age_minutes(path, now)
        if age is None or age > max_age_minutes:
            continue
        row = {
            "path": str(path),
            "name": path.name,
            "age_minutes": round(age, 2),
        }
        payload = _safe_read_json(path)
        if payload is not None:
            for key in ("issue", "issue_number", "branch", "head", "head_sha"):
                value = payload.get(key)
                if value is not None:
                    row[key] = value
            if isinstance(payload.get("idempotency_key"), str):
                row["idempotency_key"] = payload["idempotency_key"]
            if isinstance(payload.get("agent"), str):
                row["agent"] = payload["agent"]
        rows.append(row)
    return rows


def detect_fleet_coordination(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    coord_json = repo_root / ".aragora" / "fleet_coordination.json"
    coord_lock = repo_root / ".aragora" / "fleet_coordination.lock"
    payload = _safe_read_json(coord_json)
    if payload is not None:
        out["fleet_coordination_json"] = payload
    if coord_lock.exists():
        try:
            lock_text = coord_lock.read_text(encoding="utf-8").strip()
        except OSError:
            lock_text = ""
        out["fleet_coordination_lock"] = lock_text
    return out


def detect_codex_desktop_automations(
    repo_root: Path,
    *,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
) -> dict[str, Any]:
    """Invoke ``scripts/check_codex_desktop_automations.py --json`` if present."""
    script = repo_root / "scripts" / "check_codex_desktop_automations.py"
    if not script.exists():
        return {}
    python = sys.executable or shutil.which("python3") or "python3"
    try:
        proc = subprocess.run(
            [python, str(script), "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    core_writers = payload.get("core_writers") or {}
    summary = {
        "automation_count": payload.get("automation_count"),
        "core_writers": {
            name: {
                "status": (data or {}).get("status"),
                "kind": (data or {}).get("kind"),
            }
            for name, data in core_writers.items()
            if isinstance(data, dict)
        },
        "issues": payload.get("issues") or [],
    }
    return summary


def detect_codex_cli_sessions(
    codex_home: Path,
    *,
    now: dt.datetime,
    max_age_minutes: float,
    limit: int = 20,
    scan_limit: int = DEFAULT_CODEX_SESSION_SCAN_LIMIT,
) -> list[dict[str, Any]]:
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            (p for p in sessions_dir.rglob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    for path in candidates[: max(0, int(scan_limit))]:
        age = _file_age_minutes(path, now)
        if age is None or age > max_age_minutes:
            continue
        try:
            relative_path = str(path.relative_to(sessions_dir))
        except ValueError:
            relative_path = path.name
        out.append(
            {
                "path": str(path),
                "name": path.name,
                "relative_path": relative_path,
                "age_minutes": round(age, 2),
                **read_codex_session_metadata(path),
            }
        )
        if len(out) >= limit:
            break
    return out


def read_codex_session_metadata(path: Path, *, max_lines: int = 50) -> dict[str, Any]:
    """Read safe session metadata from a Codex JSONL file.

    Codex rollout files can contain full user/assistant content. This helper
    only inspects the top-level payload metadata that identifies the working
    directory and git branch, and deliberately ignores message content.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return {}
    with handle:
        for index, line in enumerate(handle):
            if index >= max_lines:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            git = payload.get("git")
            git_payload = git if isinstance(git, dict) else {}
            cwd = payload.get("cwd")
            branch = _stable_branch_name(git_payload.get("branch"))
            metadata: dict[str, Any] = {}
            if isinstance(payload.get("id"), str):
                metadata["thread_id"] = payload["id"]
            if isinstance(payload.get("source"), str):
                metadata["source"] = payload["source"]
            if isinstance(cwd, str) and cwd.strip():
                metadata["cwd"] = cwd.strip()
            if branch:
                metadata["branch"] = branch
            if isinstance(git_payload.get("commit_hash"), str):
                metadata["commit_hash"] = git_payload["commit_hash"]
            if isinstance(git_payload.get("repository_url"), str):
                metadata["repository_url"] = git_payload["repository_url"]
            if metadata:
                return metadata
    return {}


def detect_agent_process_census(
    repo_root: Path,
    *,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
) -> dict[str, Any]:
    """Invoke the merged agent-bridge process census if available."""
    script = repo_root / "scripts" / "agent_bridge.py"
    if not script.exists():
        return {}
    python = sys.executable or shutil.which("python3") or "python3"
    try:
        proc = subprocess.run(
            [python, str(script), "processes", "--json", "--summary-only"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    by_role = payload.get("by_role")
    return {
        "ok": bool(payload.get("ok")),
        "total": payload.get("total"),
        "by_role": by_role if isinstance(by_role, dict) else {},
    }


def fetch_open_prs(
    *,
    limit: int = DEFAULT_MAX_PR_FETCH,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
) -> list[dict[str, Any]]:
    """Best-effort ``gh pr list`` fetch; returns [] on any failure."""
    gh = shutil.which("gh")
    if not gh:
        return []
    try:
        proc = subprocess.run(
            [
                gh,
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                str(int(limit)),
                "--json",
                "number,title,headRefName,author,isDraft,createdAt,updatedAt,url",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "number": entry.get("number"),
                "title": entry.get("title"),
                "branch": _stable_branch_name(entry.get("headRefName")),
                "is_draft": bool(entry.get("isDraft")),
                "author": ((entry.get("author") or {}).get("login")),
                "created_at": entry.get("createdAt"),
                "updated_at": entry.get("updatedAt"),
                "url": entry.get("url"),
            }
        )
    return out


SignalBucket = dict[str, tuple[set[str], set[str]]]


def _record_signal(
    *,
    sink: dict[str, SignalBucket],
    key_kind: str,
    key_value: str | None,
    source: str,
    detail: str | None = None,
) -> None:
    if not key_value:
        return
    bucket = sink.setdefault(key_kind, {})
    sources, details = bucket.setdefault(key_value, (set(), set()))
    sources.add(source)
    if detail:
        details.add(detail)


def build_overlap_report(
    *,
    worktrees: list[dict[str, Any]],
    dispatch_contracts: list[dict[str, Any]],
    issue_claims: list[dict[str, Any]],
    automation_outbox: list[dict[str, Any]],
    codex_cli_sessions: list[dict[str, Any]],
    open_prs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-reference branches/paths/issues across signal sources."""
    signals: dict[str, SignalBucket] = {}
    for wt in worktrees:
        path = str(wt.get("path") or "")
        branch = _stable_branch_name(wt.get("branch"))
        if path:
            _record_signal(
                sink=signals,
                key_kind="worktree_path",
                key_value=path,
                source="git_worktree",
                detail=("locked" if wt.get("locked") else "active"),
            )
        if branch:
            _record_signal(
                sink=signals,
                key_kind="branch",
                key_value=branch,
                source="git_worktree",
                detail=path,
            )
    for entry in dispatch_contracts:
        branch = _stable_branch_name(entry.get("branch") or entry.get("head"))
        if branch:
            _record_signal(
                sink=signals,
                key_kind="branch",
                key_value=branch,
                source="dispatch_contract",
                detail=entry.get("name"),
            )
        if entry.get("issue") is not None or entry.get("issue_number") is not None:
            issue = entry.get("issue_number") or entry.get("issue")
            _record_signal(
                sink=signals,
                key_kind="issue",
                key_value=str(issue),
                source="dispatch_contract",
                detail=entry.get("name"),
            )
    for entry in issue_claims:
        if entry.get("issue") is not None or entry.get("issue_number") is not None:
            issue = entry.get("issue_number") or entry.get("issue")
            _record_signal(
                sink=signals,
                key_kind="issue",
                key_value=str(issue),
                source="issue_claim",
                detail=entry.get("name"),
            )
    for entry in automation_outbox:
        branch = _stable_branch_name(entry.get("branch"))
        if branch:
            _record_signal(
                sink=signals,
                key_kind="branch",
                key_value=branch,
                source="automation_outbox",
                detail=entry.get("idempotency_key"),
            )
    for entry in codex_cli_sessions:
        detail = str(entry.get("relative_path") or entry.get("name") or "codex-session")
        branch = _stable_branch_name(entry.get("branch"))
        if branch:
            _record_signal(
                sink=signals,
                key_kind="branch",
                key_value=branch,
                source="codex_cli_session",
                detail=detail,
            )
        cwd = entry.get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            _record_signal(
                sink=signals,
                key_kind="worktree_path",
                key_value=cwd.strip(),
                source="codex_cli_session",
                detail=detail,
            )
    for entry in open_prs:
        branch = _stable_branch_name(entry.get("branch"))
        if branch:
            _record_signal(
                sink=signals,
                key_kind="branch",
                key_value=branch,
                source="open_pr",
                detail=f"#{entry.get('number')}" if entry.get("number") else None,
            )

    overlaps: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for kind, bucket in signals.items():
        for value, (sources, details) in bucket.items():
            counts[kind] += 1
            if len(sources) > 1:
                overlaps.append(
                    {
                        "kind": kind,
                        "value": value,
                        "sources": sorted(sources),
                        "details": sorted(d for d in details if d),
                    }
                )
    overlaps.sort(key=lambda row: (row["kind"], row["value"]))
    return {
        "counts": dict(counts),
        "overlaps": overlaps,
        "overlap_count": len(overlaps),
    }


def build_payload(
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    codex_home: Path = DEFAULT_CODEX_HOME,
    now: dt.datetime | None = None,
    max_age_minutes: float = DEFAULT_MAX_AGE_MINUTES,
    skip_gh: bool = False,
    max_pr_fetch: int = DEFAULT_MAX_PR_FETCH,
    skip_codex_desktop: bool = False,
    skip_process_census: bool = False,
    codex_session_scan_limit: int = DEFAULT_CODEX_SESSION_SCAN_LIMIT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    codex_home = codex_home.expanduser()
    timestamp = now or _utc_now()

    worktrees = detect_worktrees(repo_root)
    aragora_dir = repo_root / ".aragora"
    dispatch_contracts = detect_recent_jsonl_files(
        aragora_dir / "dispatch_contracts",
        now=timestamp,
        max_age_minutes=max_age_minutes,
    )
    issue_claims = detect_recent_jsonl_files(
        aragora_dir / "issue_claims",
        now=timestamp,
        max_age_minutes=max_age_minutes,
    )
    work_leases = detect_recent_jsonl_files(
        aragora_dir / "work-leases",
        now=timestamp,
        max_age_minutes=max_age_minutes,
    )
    automation_outbox = detect_recent_jsonl_files(
        aragora_dir / "automation-outbox",
        now=timestamp,
        max_age_minutes=max_age_minutes,
    )
    fleet = detect_fleet_coordination(repo_root)
    codex_desktop: dict[str, Any] = {}
    if not skip_codex_desktop:
        codex_desktop = detect_codex_desktop_automations(repo_root)
    codex_cli_sessions = detect_codex_cli_sessions(
        codex_home,
        now=timestamp,
        max_age_minutes=max_age_minutes,
        scan_limit=codex_session_scan_limit,
    )
    process_census: dict[str, Any] = {}
    if not skip_process_census:
        process_census = detect_agent_process_census(repo_root)
    open_prs: list[dict[str, Any]] = []
    if not skip_gh:
        open_prs = fetch_open_prs(limit=max_pr_fetch)

    overlap = build_overlap_report(
        worktrees=worktrees,
        dispatch_contracts=dispatch_contracts,
        issue_claims=issue_claims,
        automation_outbox=automation_outbox,
        codex_cli_sessions=codex_cli_sessions,
        open_prs=open_prs,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _isoformat(timestamp),
        "repo_root": str(repo_root),
        "codex_home": str(codex_home),
        "max_age_minutes": max_age_minutes,
        "codex_session_scan_limit": codex_session_scan_limit,
        "skip_gh": skip_gh,
        "skip_codex_desktop": skip_codex_desktop,
        "skip_process_census": skip_process_census,
        "worktrees": worktrees,
        "dispatch_contracts": dispatch_contracts,
        "issue_claims": issue_claims,
        "work_leases": work_leases,
        "automation_outbox": automation_outbox,
        "fleet_coordination": fleet,
        "codex_desktop_automations": codex_desktop,
        "codex_cli_sessions": codex_cli_sessions,
        "process_census": process_census,
        "open_prs": open_prs,
        "overlap_report": overlap,
    }
    return payload


def render_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Active agent sessions (generated_at={payload.get('generated_at')})")
    lines.append(f"  repo_root: {payload.get('repo_root')}")
    lines.append(f"  codex_home: {payload.get('codex_home')}")
    lines.append("")

    worktrees = payload.get("worktrees") or []
    lines.append(f"git worktrees ({len(worktrees)}):")
    for wt in worktrees[:30]:
        flag = " [LOCKED]" if wt.get("locked") else ""
        lines.append(f"  - {wt.get('branch') or '(detached)'}{flag}  {wt.get('path', '')}")
    if len(worktrees) > 30:
        lines.append(f"  ... ({len(worktrees) - 30} more)")
    lines.append("")

    for label, key in (
        ("dispatch contracts", "dispatch_contracts"),
        ("issue claims", "issue_claims"),
        ("work leases", "work_leases"),
        ("automation outbox", "automation_outbox"),
        ("codex CLI sessions", "codex_cli_sessions"),
    ):
        rows = payload.get(key) or []
        lines.append(f"{label} (recent within max_age, {len(rows)}):")
        for row in rows[:10]:
            issue = row.get("issue") or row.get("issue_number")
            branch = row.get("branch") or "-"
            age = row.get("age_minutes")
            name = row.get("relative_path") or row.get("name") or "?"
            lines.append(f"  - {name}  issue={issue}  branch={branch}  age_min={age}")
        if len(rows) > 10:
            lines.append(f"  ... ({len(rows) - 10} more)")
        lines.append("")

    codex_desktop = payload.get("codex_desktop_automations") or {}
    core_writers = codex_desktop.get("core_writers") or {}
    if core_writers:
        lines.append("codex desktop core writers:")
        for name, info in core_writers.items():
            lines.append(f"  - {name}: status={info.get('status')}")
        lines.append("")

    process_census = payload.get("process_census") or {}
    if process_census:
        lines.append(
            f"agent process census: ok={process_census.get('ok')} "
            f"total={process_census.get('total')}"
        )
        by_role = process_census.get("by_role") or {}
        for role, count in sorted(by_role.items()):
            lines.append(f"  - {role}: {count}")
        lines.append("")

    prs = payload.get("open_prs") or []
    lines.append(f"open PRs ({len(prs)}):")
    for pr in prs[:20]:
        draft = "DRAFT" if pr.get("is_draft") else "READY"
        lines.append(
            f"  - #{pr.get('number')} {draft}  branch={pr.get('branch')}  "
            f"author={pr.get('author')}  {(pr.get('title') or '')[:65]}"
        )
    if len(prs) > 20:
        lines.append(f"  ... ({len(prs) - 20} more)")
    lines.append("")

    overlap = payload.get("overlap_report") or {}
    lines.append(f"overlap report (count={overlap.get('overlap_count', 0)}):")
    for ov in (overlap.get("overlaps") or [])[:20]:
        sources = "+".join(ov.get("sources") or [])
        lines.append(f"  - {ov.get('kind')}={ov.get('value')}  sources=[{sources}]")
    if not overlap.get("overlaps"):
        lines.append("  (no cross-source overlaps detected)")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=f"Path to the repository root (default: {DEFAULT_REPO_ROOT})",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=DEFAULT_CODEX_HOME,
        help=f"Path to ~/.codex (default: {DEFAULT_CODEX_HOME})",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=DEFAULT_MAX_AGE_MINUTES,
        help="Skip JSON/JSONL files older than this many minutes "
        f"(default: {DEFAULT_MAX_AGE_MINUTES})",
    )
    parser.add_argument(
        "--max-pr-fetch",
        type=int,
        default=DEFAULT_MAX_PR_FETCH,
        help=f"Max open PRs to fetch via gh (default: {DEFAULT_MAX_PR_FETCH})",
    )
    parser.add_argument(
        "--skip-gh",
        action="store_true",
        help="Skip the gh pr list invocation.",
    )
    parser.add_argument(
        "--skip-codex-desktop",
        action="store_true",
        help="Skip the Codex Desktop automation status subprocess.",
    )
    parser.add_argument(
        "--skip-process-census",
        action="store_true",
        help="Skip the agent_bridge.py process census subprocess.",
    )
    parser.add_argument(
        "--codex-session-scan-limit",
        type=int,
        default=DEFAULT_CODEX_SESSION_SCAN_LIMIT,
        help="Max nested Codex CLI session files to inspect after mtime sorting "
        f"(default: {DEFAULT_CODEX_SESSION_SCAN_LIMIT})",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(
        repo_root=args.repo_root,
        codex_home=args.codex_home,
        max_age_minutes=float(args.max_age_minutes),
        skip_gh=bool(args.skip_gh),
        max_pr_fetch=int(args.max_pr_fetch),
        skip_codex_desktop=bool(args.skip_codex_desktop),
        skip_process_census=bool(args.skip_process_census),
        codex_session_scan_limit=int(args.codex_session_scan_limit),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(render_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
