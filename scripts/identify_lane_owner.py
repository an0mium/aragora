#!/usr/bin/env python3
"""Read-only consolidator: lane id / PR / branch / worktree → owner identity.

Implements Phase A of the agent-steering primitive plan. Walks the
existing aragora signals to answer the question every operator hits
when a fan-out lane is stuck: *who actually owns this PR, and where is
their session running?*

Lookup sources (read-only, in this precedence):

  1. ``.aragora/agent-bridge/lanes.json`` — primary owner identity,
     from ``LaneRecord`` rows written by
     ``scripts/claim_active_agent_lane.py``. Carries:
     ``owner_session``, ``source``, ``branch``, ``worktree``,
     ``pr_number``, plus the optional richer identity fields
     ``codex_thread_id``, ``codex_rollout_path``, ``desktop_label``,
     ``session_title`` when the claimer supplied them.

  2. ``scripts/agent_bridge.py operator-snapshot --json``
     ``process_census`` — best-effort live PID lookup when the
     snapshot exposes cwd-bearing process records. If the snapshot
     only exposes role counts, this fails closed with an explicit
     reason instead of guessing.

  3. ``~/.codex/sessions/**/*.jsonl`` — exact match via the lane's
     recorded ``codex_rollout_path`` or ``codex_thread_id``, with a
     fuzzy fallback: any recent rollout whose body contains the
     lane's worktree path string.

  4. ``~/.claude/projects/<encoded-cwd>/<uuid>.jsonl`` — Claude Code
     session lookup via the standard cwd-encoding rule
     (``/`` → ``-``).

  5. ``~/.factory/background-processes.json`` — Factory Droid
     background session match by branch or worktree.

  6. ``.aragora/operator-steering/<owner_session>/`` — pending
     steering-message inbox count (Phase B-built dir; Phase A only
     reads).

Pure stdlib. No ``aragora.*`` imports. Read-only — never mutates
GitHub state, lane registry, mailboxes, or any other on-disk file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths (overridable for tests)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
LANE_REGISTRY_DEFAULT = REPO_ROOT / ".aragora" / "agent-bridge" / "lanes.json"
STEERING_INBOX_ROOT_DEFAULT = REPO_ROOT / ".aragora" / "operator-steering"
CODEX_SESSIONS_ROOT_DEFAULT = Path.home() / ".codex" / "sessions"
CLAUDE_PROJECTS_ROOT_DEFAULT = Path.home() / ".claude" / "projects"
FACTORY_BG_PROCESSES_DEFAULT = Path.home() / ".factory" / "background-processes.json"

# Fuzzy codex rollout search window (seconds).
CODEX_FUZZY_MAX_AGE_SECONDS = 4 * 60 * 60
ACTIVE_STATUSES = {"active", "running", "pending", "queued", "claimed"}
CONFLICT_STATUSES = {"conflict", "conflicting"}
COMPLETED_STATUSES = {"completed", "released"}

# Subprocess timeout for ``agent_bridge operator-snapshot``.
SNAPSHOT_TIMEOUT_SECONDS = 30

SnapshotProvider = Callable[[], dict[str, Any] | None]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LaneOwnerInfo:
    """Consolidated owner identity for one lane.

    Mirrors the schema documented in the agent-steering plan. Fields
    that aren't applicable to a given lane carry empty dicts (with a
    ``reason`` for debuggability) rather than ``None`` so JSON
    consumers can switch on ``found``.
    """

    lane_id: str
    owner_session: str
    source: str
    status: str
    branch: str | None
    worktree: str | None
    pr_number: int | None
    goal: str | None
    updated_at: str | None
    codex_thread_id: str | None
    codex_rollout_path: str | None
    desktop_label: str | None
    session_title: str | None
    live_process: dict[str, Any]
    codex_thread: dict[str, Any]
    claude_session: dict[str, Any]
    factory_droid: dict[str, Any]
    steering_inbox_path: str
    pending_message_count: int
    dispatchable: bool
    dispatch_blocker: str | None
    steering_command: str | None
    harness_confidence: str


# ---------------------------------------------------------------------------
# Registry loading + match
# ---------------------------------------------------------------------------


def load_lane_records(registry_path: Path = LANE_REGISTRY_DEFAULT) -> list[dict[str, Any]]:
    """Read the lane registry; return ``[]`` on missing / unparseable."""

    if not registry_path.exists():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _status_rank(raw_status: Any) -> int:
    """Rank lane statuses for non-unique selectors; lower is preferred."""

    status = str(raw_status or "").strip().lower()
    if status in ACTIVE_STATUSES:
        return 0
    if status in CONFLICT_STATUSES:
        return 1
    if status in COMPLETED_STATUSES:
        return 2
    return 3


def _updated_at_timestamp(raw_updated_at: Any) -> float:
    """Parse ``updated_at`` for ordering; invalid or missing values sort oldest."""

    text = str(raw_updated_at or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _best_lane_match(matches: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the best row for non-unique selectors like PR, branch, or worktree."""

    if not matches:
        return None
    indexed = enumerate(matches)
    _, best = min(
        indexed,
        key=lambda item: (
            _status_rank(item[1].get("status")),
            -_updated_at_timestamp(item[1].get("updated_at")),
            item[0],
        ),
    )
    return best


def find_lane(
    records: Sequence[dict[str, Any]],
    *,
    lane_id: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
    worktree: str | None = None,
) -> dict[str, Any] | None:
    """Return the best matching lane record by lane_id > pr > branch > worktree.

    Multiple historical rows can target the same PR/branch/worktree. Prefer an
    active row, then a conflict row, then the most recently updated completed or
    released row, so owner lookup does not silently route an operator to a stale
    completed lane. Exact lane-id lookup preserves registry order.
    """

    if lane_id:
        return next((r for r in records if r.get("lane_id") == lane_id), None)
    if pr is not None:
        matches = []
        for r in records:
            try:
                if int(r.get("pr_number") or 0) == int(pr):
                    matches.append(r)
            except (TypeError, ValueError):
                continue
        return _best_lane_match(matches)
    if branch:
        return _best_lane_match([r for r in records if r.get("branch") == branch])
    if worktree:
        wt_norm = os.path.normpath(worktree)
        matches = []
        for r in records:
            rwt = r.get("worktree")
            if rwt and os.path.normpath(rwt) == wt_norm:
                matches.append(r)
        return _best_lane_match(matches)
    return None


# ---------------------------------------------------------------------------
# Live process lookup (via agent_bridge operator-snapshot subprocess)
# ---------------------------------------------------------------------------


def _default_snapshot_provider() -> dict[str, Any] | None:
    """Shell out to ``scripts/agent_bridge.py operator-snapshot --json``."""

    bridge = REPO_ROOT / "scripts" / "agent_bridge.py"
    if not bridge.is_file():
        return None
    try:
        res = subprocess.run(
            [sys.executable, str(bridge), "operator-snapshot", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SNAPSHOT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        out = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


_SOURCE_ROLE_MAP: dict[str, tuple[str, ...]] = {
    "claude": ("claude_code",),
    "claude_code": ("claude_code",),
    "codex": ("codex_cli", "codex_app_server"),
    "codex_cli": ("codex_cli",),
    "codex_app": ("codex_app_server",),
    "codex_app_server": ("codex_app_server",),
    "droid": ("factory_droid",),
    "factory": ("factory_droid",),
    "factory_droid": ("factory_droid",),
}


def _family_hints_for_lane(lane: dict[str, Any]) -> tuple[str, ...]:
    """Return live-process role hints implied by lane metadata."""

    hints: list[str] = []
    for raw in (
        lane.get("source"),
        lane.get("owner_session"),
        lane.get("lane_id"),
        lane.get("branch"),
    ):
        text = str(raw or "").lower()
        if not text:
            continue
        for token, roles in _SOURCE_ROLE_MAP.items():
            if token in text:
                for role in roles:
                    if role not in hints:
                        hints.append(role)
    return tuple(hints)


def _process_cwd(item: dict[str, Any]) -> str:
    """Return the cwd-like field from a process record, if available."""

    raw = item.get("cwd") or item.get("worktree")
    return str(raw) if raw else ""


def _process_match_payload(role: str, item: dict[str, Any], cwd: str) -> dict[str, Any]:
    """Safe metadata for a live process matched by cwd."""

    return {"pid": item.get("pid"), "family": role, "cwd": cwd}


def _collect_process_matches(
    process_census: dict[str, Any],
    *,
    target_norm: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Collect cwd-matching process records from the operator snapshot.

    The live ``operator-snapshot`` contract currently reports
    ``by_role`` as role counts and carries process rows in
    ``records``. Older fixtures used ``by_role`` as role -> process
    rows; keep that as a compatibility fallback, but do not require it.
    The boolean reports whether any cwd-bearing record existed at all.
    """

    matches: list[dict[str, Any]] = []
    saw_cwd_bearing_record = False

    records = process_census.get("records", [])
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            cwd = _process_cwd(item)
            if not cwd:
                continue
            saw_cwd_bearing_record = True
            if os.path.normpath(cwd) == target_norm:
                matches.append(
                    _process_match_payload(str(item.get("role") or "unknown"), item, cwd)
                )

    by_role = process_census.get("by_role", {})
    if isinstance(by_role, dict):
        for role, items in by_role.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                cwd = _process_cwd(item)
                if not cwd:
                    continue
                saw_cwd_bearing_record = True
                if os.path.normpath(cwd) == target_norm:
                    matches.append(_process_match_payload(str(role), item, cwd))

    matches.sort(key=lambda m: (str(m.get("family") or ""), str(m.get("pid") or "")))
    return matches, saw_cwd_bearing_record


def lookup_live_process(
    lane: dict[str, Any],
    *,
    snapshot_provider: SnapshotProvider | None = None,
) -> dict[str, Any]:
    """Best-effort PID lookup: match cwd-bearing snapshot records to lane.worktree."""

    target_wt = lane.get("worktree") or ""
    if not target_wt:
        return {"found": False, "reason": "lane has no worktree to match against"}
    target_norm = os.path.normpath(target_wt)

    provider = snapshot_provider or _default_snapshot_provider
    snap = provider()
    if snap is None:
        return {"found": False, "reason": "operator-snapshot unavailable"}

    process_census = snap.get("process_census", {})
    if not isinstance(process_census, dict):
        return {"found": False, "reason": "operator-snapshot has no process_census object"}

    matches, saw_cwd_bearing_record = _collect_process_matches(
        process_census,
        target_norm=target_norm,
    )
    if len(matches) == 1:
        match = matches[0]
        return {
            "found": True,
            "pid": match.get("pid"),
            "family": match.get("family"),
            "cwd": match.get("cwd"),
            "matched_via": "lane.worktree ↔ process_census.cwd (exact)",
        }
    if len(matches) > 1:
        family_hints = _family_hints_for_lane(lane)
        hinted = [m for m in matches if m.get("family") in family_hints]
        if len(hinted) == 1:
            match = hinted[0]
            return {
                "found": True,
                "pid": match.get("pid"),
                "family": match.get("family"),
                "cwd": match.get("cwd"),
                "matched_via": (
                    "lane.worktree ↔ process_census.cwd (exact; "
                    "disambiguated by lane family metadata)"
                ),
            }
        if hinted:
            reason = (
                "ambiguous_same_worktree: multiple process_census entries matched "
                f"{target_norm}; lane family hints {list(family_hints)} still matched "
                f"{len(hinted)} entries"
            )
        elif family_hints:
            reason = (
                "ambiguous_same_worktree: multiple process_census entries matched "
                f"{target_norm}; none matched lane family hints {list(family_hints)}"
            )
        else:
            reason = (
                "ambiguous_same_worktree: multiple process_census entries matched "
                f"{target_norm}; no lane family metadata available to disambiguate"
            )
        return {"found": False, "reason": reason, "matches": matches}
    if not saw_cwd_bearing_record:
        return {
            "found": False,
            "reason": (
                "operator-snapshot process_census has no cwd-bearing process records; "
                f"cannot match lane worktree {target_norm}"
            ),
        }
    return {
        "found": False,
        "reason": f"no process_census entry matched worktree {target_norm}",
    }


# ---------------------------------------------------------------------------
# Codex thread lookup
# ---------------------------------------------------------------------------


_ROLLOUT_FILENAME_RE = re.compile(
    r"rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-([0-9a-f-]+)\.jsonl$"
)


def _extract_thread_id(rollout_filename: str) -> str:
    m = _ROLLOUT_FILENAME_RE.search(rollout_filename)
    return m.group(1) if m else ""


def lookup_codex_thread(
    lane: dict[str, Any],
    *,
    sessions_root: Path = CODEX_SESSIONS_ROOT_DEFAULT,
    fuzzy_max_age_seconds: int = CODEX_FUZZY_MAX_AGE_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    """Find the Codex rollout file backing this lane.

    Tries (in order): exact ``codex_rollout_path`` from the lane,
    exact ``codex_thread_id`` against rollout filenames, and a fuzzy
    fallback that scans recently-modified rollouts for the lane's
    worktree path appearing in the rollout body.
    """

    if not sessions_root.is_dir():
        return {"found": False, "reason": f"codex sessions root absent ({sessions_root})"}

    rollout_path_hint = lane.get("codex_rollout_path")
    if rollout_path_hint:
        p = Path(os.path.expanduser(str(rollout_path_hint)))
        if p.is_file():
            return {
                "found": True,
                "thread_id": _extract_thread_id(p.name),
                "rollout_path": str(p),
                "mtime": p.stat().st_mtime,
                "matched_via": "lane.codex_rollout_path (exact)",
            }

    thread_id_hint = lane.get("codex_thread_id")
    if thread_id_hint:
        for p in sessions_root.rglob("*.jsonl"):
            if str(thread_id_hint) in p.name:
                return {
                    "found": True,
                    "thread_id": str(thread_id_hint),
                    "rollout_path": str(p),
                    "mtime": p.stat().st_mtime,
                    "matched_via": "lane.codex_thread_id (exact filename match)",
                }

    # Fuzzy: scan recent rollouts for the lane's worktree string.
    worktree = lane.get("worktree")
    if not worktree:
        return {"found": False, "reason": "no codex identity hint and no worktree to fuzzy-match"}

    current = now if now is not None else time.time()
    candidates: list[tuple[float, Path]] = []
    for p in sessions_root.rglob("*.jsonl"):
        try:
            st = p.stat()
        except OSError:
            continue
        if current - st.st_mtime > fuzzy_max_age_seconds:
            continue
        candidates.append((st.st_mtime, p))

    target_str = str(worktree)
    matches: list[tuple[float, Path]] = []
    for mtime, p in candidates:
        try:
            # Cheap substring scan; rollouts are JSONL but the cwd
            # appears as a literal path string in numerous event
            # payloads when Codex tools fire.
            if target_str in p.read_text(encoding="utf-8", errors="ignore"):
                matches.append((mtime, p))
        except OSError:
            continue

    if not matches:
        return {
            "found": False,
            "reason": (
                f"no recent codex rollout (within {fuzzy_max_age_seconds // 60}m) "
                f"contained worktree string"
            ),
        }
    if len(matches) > 1:
        # Multiple matches — return the most-recent but flag the
        # ambiguity so the operator knows the answer isn't unique.
        matches.sort(key=lambda t: t[0], reverse=True)
        latest = matches[0][1]
        return {
            "found": True,
            "thread_id": _extract_thread_id(latest.name),
            "rollout_path": str(latest),
            "mtime": matches[0][0],
            "matched_via": (
                f"fuzzy: worktree string found in {len(matches)} recent "
                "rollouts; returning most-recent (ambiguous)"
            ),
        }
    mtime, p = matches[0]
    return {
        "found": True,
        "thread_id": _extract_thread_id(p.name),
        "rollout_path": str(p),
        "mtime": mtime,
        "matched_via": "fuzzy: worktree string found in single recent rollout",
    }


# ---------------------------------------------------------------------------
# Claude Code session lookup
# ---------------------------------------------------------------------------


def _encode_cwd_for_claude(cwd: str) -> str:
    """Replicate Claude Code's project-dir encoding rule (``/`` → ``-``)."""

    # Drop trailing slash for stable encoding.
    cwd_clean = cwd.rstrip("/")
    encoded = cwd_clean.replace("/", "-")
    # Leading slash → leading dash; Claude's encoding starts with '-'.
    if not encoded.startswith("-"):
        encoded = "-" + encoded
    return encoded


def lookup_claude_session(
    lane: dict[str, Any],
    *,
    projects_root: Path = CLAUDE_PROJECTS_ROOT_DEFAULT,
) -> dict[str, Any]:
    """Find the Claude Code session backing this lane (best-effort)."""

    worktree = lane.get("worktree")
    if not worktree:
        return {"found": False, "reason": "lane has no worktree to match against"}
    if not projects_root.is_dir():
        return {"found": False, "reason": f"claude projects root absent ({projects_root})"}

    encoded = _encode_cwd_for_claude(str(worktree))
    candidate = projects_root / encoded
    if not candidate.is_dir():
        return {"found": False, "reason": f"no claude project dir matched encoding ({encoded})"}

    sessions = sorted(
        (p for p in candidate.glob("*.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        return {
            "found": False,
            "reason": f"claude project dir {encoded} has no .jsonl session files",
        }
    latest = sessions[0]
    return {
        "found": True,
        "session_uuid": latest.stem,
        "transcript_path": str(latest),
        "mtime": latest.stat().st_mtime,
        "matched_via": "lane.worktree → claude project encoding (most-recent .jsonl)",
    }


# ---------------------------------------------------------------------------
# Factory Droid lookup
# ---------------------------------------------------------------------------


def lookup_factory_droid(
    lane: dict[str, Any],
    *,
    bg_path: Path = FACTORY_BG_PROCESSES_DEFAULT,
) -> dict[str, Any]:
    """Match the lane to a Factory Droid background-process record by branch or worktree."""

    if not bg_path.is_file():
        return {"found": False, "reason": f"factory bg-processes file absent ({bg_path})"}
    try:
        data = json.loads(bg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"found": False, "reason": "factory bg-processes file unparseable"}

    processes_raw: Any
    if isinstance(data, list):
        processes_raw = data
    elif isinstance(data, dict):
        processes_raw = data.get("processes") or data.get("background_processes") or []
    else:
        processes_raw = []

    branch = lane.get("branch")
    worktree = lane.get("worktree")
    wt_norm = os.path.normpath(str(worktree)) if worktree else None

    for p in processes_raw:
        if not isinstance(p, dict):
            continue
        if branch and p.get("branch") == branch:
            return {
                "found": True,
                "process_id": p.get("id") or p.get("pid") or p.get("session_id"),
                "branch": branch,
                "matched_via": "factory.branch (exact)",
            }
        if wt_norm:
            p_wt = p.get("worktree") or p.get("cwd") or ""
            if p_wt and os.path.normpath(str(p_wt)) == wt_norm:
                return {
                    "found": True,
                    "process_id": p.get("id") or p.get("pid") or p.get("session_id"),
                    "worktree": p_wt,
                    "matched_via": "factory.worktree (exact)",
                }
    return {"found": False, "reason": "no factory droid process matched branch or worktree"}


# ---------------------------------------------------------------------------
# Steering inbox count
# ---------------------------------------------------------------------------


def steering_inbox_for(
    owner_session: str, *, root: Path = STEERING_INBOX_ROOT_DEFAULT
) -> tuple[Path, int]:
    """Return ``(inbox_path, pending_count)``; missing dir → ``(path, 0)``."""

    inbox = root / owner_session
    if not inbox.is_dir():
        return inbox, 0
    count = sum(1 for _ in inbox.glob("*.json"))
    return inbox, count


def _dispatch_blocker_for(lane: dict[str, Any], owner_session: str) -> str | None:
    status = str(lane.get("status") or "").strip().lower()
    if not owner_session:
        return "lane has no owner_session"
    if status in ACTIVE_STATUSES:
        return None
    if status in CONFLICT_STATUSES:
        return "lane status is conflict; resolve the conflict before steering"
    if status in COMPLETED_STATUSES:
        return f"lane status is {status}; claim an active lane before steering"
    return f"lane status is {status or 'unknown'}; claim an active lane before steering"


def _steering_command_for(lane: dict[str, Any], owner_session: str) -> str | None:
    if _dispatch_blocker_for(lane, owner_session) is not None:
        return None
    parts = [
        "python3",
        "scripts/send_operator_steering.py",
        "--to",
        owner_session,
    ]
    lane_id = str(lane.get("lane_id") or "")
    if lane_id:
        parts.extend(["--lane-id", lane_id])
    raw_pr = lane.get("pr_number")
    if raw_pr is not None:
        parts.extend(["--pr", str(raw_pr)])
    parts.extend(["--priority", "blocking", "--body", "'<message>'"])
    return " ".join(parts)


def _harness_confidence_for(
    lane: dict[str, Any],
    *,
    live: dict[str, Any],
    codex: dict[str, Any],
    claude: dict[str, Any],
    factory: dict[str, Any],
) -> str:
    if any(
        lane.get(field)
        for field in (
            "codex_thread_id",
            "codex_rollout_path",
            "desktop_label",
            "session_title",
        )
    ):
        return "recorded_identity"
    if live.get("found"):
        return "live_process"
    if codex.get("found"):
        matched_via = str(codex.get("matched_via") or "")
        if "ambiguous" in matched_via or "fuzzy" in matched_via:
            return "mailbox_only_fuzzy_thread"
        return "codex_thread_best_effort"
    if claude.get("found"):
        return "claude_session_best_effort"
    if factory.get("found"):
        return "factory_droid_best_effort"
    return "mailbox_only"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def build_owner_info(
    lane: dict[str, Any],
    *,
    snapshot_provider: SnapshotProvider | None = None,
    sessions_root: Path = CODEX_SESSIONS_ROOT_DEFAULT,
    projects_root: Path = CLAUDE_PROJECTS_ROOT_DEFAULT,
    bg_path: Path = FACTORY_BG_PROCESSES_DEFAULT,
    steering_inbox_root: Path = STEERING_INBOX_ROOT_DEFAULT,
    fuzzy_now: float | None = None,
) -> LaneOwnerInfo:
    owner = str(lane.get("owner_session") or "")
    live = lookup_live_process(lane, snapshot_provider=snapshot_provider)
    codex = lookup_codex_thread(lane, sessions_root=sessions_root, now=fuzzy_now)
    claude = lookup_claude_session(lane, projects_root=projects_root)
    factory = lookup_factory_droid(lane, bg_path=bg_path)
    inbox_path, pending = steering_inbox_for(owner, root=steering_inbox_root)
    dispatch_blocker = _dispatch_blocker_for(lane, owner)

    raw_pr = lane.get("pr_number")
    try:
        pr_number = int(raw_pr) if raw_pr is not None else None
    except (TypeError, ValueError):
        pr_number = None

    return LaneOwnerInfo(
        lane_id=str(lane.get("lane_id") or ""),
        owner_session=owner,
        source=str(lane.get("source") or ""),
        status=str(lane.get("status") or ""),
        branch=lane.get("branch"),
        worktree=lane.get("worktree"),
        pr_number=pr_number,
        goal=lane.get("goal"),
        updated_at=lane.get("updated_at"),
        codex_thread_id=lane.get("codex_thread_id"),
        codex_rollout_path=lane.get("codex_rollout_path"),
        desktop_label=lane.get("desktop_label"),
        session_title=lane.get("session_title"),
        live_process=live,
        codex_thread=codex,
        claude_session=claude,
        factory_droid=factory,
        steering_inbox_path=str(inbox_path),
        pending_message_count=pending,
        dispatchable=dispatch_blocker is None,
        dispatch_blocker=dispatch_blocker,
        steering_command=_steering_command_for(lane, owner),
        harness_confidence=_harness_confidence_for(
            lane,
            live=live,
            codex=codex,
            claude=claude,
            factory=factory,
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _glyph(found: bool) -> str:
    return "✓" if found else "✗"


def _print_human(info: LaneOwnerInfo) -> None:
    print(f"lane_id:        {info.lane_id}")
    print(f"owner_session:  {info.owner_session or '(none)'}")
    print(f"source:         {info.source or '(unspecified)'}")
    print(f"status:         {info.status or '(unspecified)'}")
    print(f"branch:         {info.branch or '-'}")
    print(f"worktree:       {info.worktree or '-'}")
    print(f"pr_number:      {info.pr_number if info.pr_number is not None else '-'}")
    print(f"goal:           {info.goal or '-'}")
    print(f"updated_at:     {info.updated_at or '-'}")
    print()
    print("self-supplied identity fields:")
    print(f"  codex_thread_id:    {info.codex_thread_id or '(not supplied)'}")
    print(f"  codex_rollout_path: {info.codex_rollout_path or '(not supplied)'}")
    print(f"  desktop_label:      {info.desktop_label or '(not supplied)'}")
    print(f"  session_title:      {info.session_title or '(not supplied)'}")
    print()
    print("best-effort live lookups:")
    print(f"  live_process:   {_glyph(info.live_process.get('found', False))}  {info.live_process}")
    print(f"  codex_thread:   {_glyph(info.codex_thread.get('found', False))}  {info.codex_thread}")
    print(
        f"  claude_session: {_glyph(info.claude_session.get('found', False))}  {info.claude_session}"
    )
    print(
        f"  factory_droid:  {_glyph(info.factory_droid.get('found', False))}  {info.factory_droid}"
    )
    print()
    print(f"steering_inbox_path:   {info.steering_inbox_path}")
    print(f"pending_message_count: {info.pending_message_count}")
    print(f"dispatchable:          {info.dispatchable}")
    print(f"dispatch_blocker:      {info.dispatch_blocker or '-'}")
    print(f"steering_command:      {info.steering_command or '-'}")
    print(f"harness_confidence:    {info.harness_confidence}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="identify_lane_owner.py",
        description=(
            "Read-only consolidator that answers 'who owns this lane?' by "
            "joining the agent_bridge lane registry with live process, "
            "Codex rollout, Claude project, and Factory Droid signals."
        ),
    )
    p.add_argument("--lane-id", help="Exact match on LaneRecord.lane_id.")
    p.add_argument("--pr", type=int, help="Match on LaneRecord.pr_number.")
    p.add_argument("--branch", help="Exact match on LaneRecord.branch.")
    p.add_argument("--worktree", help="Exact match on LaneRecord.worktree (path-normalised).")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human table.")
    p.add_argument(
        "--registry-path",
        type=Path,
        default=LANE_REGISTRY_DEFAULT,
        help="Override path to lanes.json (used by tests).",
    )
    p.add_argument(
        "--codex-sessions-root",
        type=Path,
        default=CODEX_SESSIONS_ROOT_DEFAULT,
        help="Override path to ~/.codex/sessions (used by tests).",
    )
    p.add_argument(
        "--claude-projects-root",
        type=Path,
        default=CLAUDE_PROJECTS_ROOT_DEFAULT,
        help="Override path to ~/.claude/projects (used by tests).",
    )
    p.add_argument(
        "--factory-bg-path",
        type=Path,
        default=FACTORY_BG_PROCESSES_DEFAULT,
        help="Override path to ~/.factory/background-processes.json (used by tests).",
    )
    p.add_argument(
        "--steering-inbox-root",
        type=Path,
        default=STEERING_INBOX_ROOT_DEFAULT,
        help="Override path to .aragora/operator-steering (used by tests).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not any([args.lane_id, args.pr is not None, args.branch, args.worktree]):
        print(
            "ERROR: provide at least one of --lane-id / --pr / --branch / --worktree",
            file=sys.stderr,
        )
        return 2

    records = load_lane_records(args.registry_path)
    if not records:
        print(
            f"ERROR: lane registry empty or missing at {args.registry_path}",
            file=sys.stderr,
        )
        return 2

    lane = find_lane(
        records,
        lane_id=args.lane_id,
        pr=args.pr,
        branch=args.branch,
        worktree=args.worktree,
    )
    if lane is None:
        criteria = {
            k: v
            for k, v in {
                "lane_id": args.lane_id,
                "pr": args.pr,
                "branch": args.branch,
                "worktree": args.worktree,
            }.items()
            if v
        }
        print(f"ERROR: no lane matched criteria {criteria}", file=sys.stderr)
        return 1

    info = build_owner_info(
        lane,
        sessions_root=args.codex_sessions_root,
        projects_root=args.claude_projects_root,
        bg_path=args.factory_bg_path,
        steering_inbox_root=args.steering_inbox_root,
    )

    if args.json:
        print(json.dumps(dataclasses.asdict(info), indent=2, sort_keys=True))
    else:
        _print_human(info)

    return 0


if __name__ == "__main__":
    sys.exit(main())
