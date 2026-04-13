#!/usr/bin/env python3
"""Passive supervisor loop for the local agent bridge.

The supervisor is intentionally read-only. It inspects bridge session state,
lane ownership, and pull request truth when available, then emits a bounded
next action for each active lane.

Usage:
  python3 scripts/agent_bridge_supervise.py --once
  python3 scripts/agent_bridge_supervise.py --once --json
  python3 scripts/agent_bridge_supervise.py --interval 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import agent_bridge  # type: ignore[import-not-found]
except ModuleNotFoundError:
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import agent_bridge  # type: ignore[import-not-found]

_APPROVAL_MARKERS = (
    "approve",
    "approval",
    "permission",
    "allow command",
    "press y",
    "type y",
    "confirm by typing",
)
_WAIT_MARKERS = (
    "waiting for ci",
    "wait for ci",
    "checks pending",
    "waiting on checks",
    "github actions",
    "buildkite",
)
_BLOCKED_MARKERS = ("blocked", "stuck", "needs human", "conflict")
_REVIEW_MARKERS = ("ready for review", "re-review", "green", "merge when green")


@dataclass(slots=True)
class WorktreeStatus:
    state: str
    dirty: bool = False
    ahead: int = 0
    behind: int = 0
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PRTruth:
    branch: str
    number: int
    url: str = ""
    is_draft: bool = False
    checks_bucket: str = "unknown"
    checks: list[dict[str, str]] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class LaneDecision:
    lane_id: str
    owner_session: str
    status: str
    next_action: str
    reason: str
    source: str = ""
    branch: str = ""
    worktree: str = ""
    pr_number: int | None = None
    pr_url: str = ""
    pr_checks_bucket: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value not in ("", None, [])}


@dataclass(slots=True)
class SupervisorSnapshot:
    generated_at: str
    decisions: list[LaneDecision]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "warnings": list(self.warnings),
            "lanes": [decision.to_dict() for decision in self.decisions],
        }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in markers)


def _classify_session_signal(text: str) -> str | None:
    if _matches_any(text, _APPROVAL_MARKERS):
        return "approve_prompt"
    if _matches_any(text, _WAIT_MARKERS):
        return "wait_for_ci"
    if _matches_any(text, _BLOCKED_MARKERS):
        return "blocked"
    if _matches_any(text, _REVIEW_MARKERS):
        return "ready_for_review"
    return None


def _session_text(session: agent_bridge.Session | None) -> str:
    if session is None:
        return ""
    lines = [session.summary] if session.summary else []
    try:
        lines.extend(agent_bridge._read_tmux_log(session.name, 5))  # noqa: SLF001
    except OSError:
        pass
    return "\n".join(line for line in lines if line)


def _run_git(worktree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _inspect_worktree(worktree_path: str) -> WorktreeStatus:
    if not worktree_path:
        return WorktreeStatus(state="missing", evidence=["worktree path missing"])

    worktree = Path(worktree_path)
    if not worktree.exists():
        return WorktreeStatus(state="missing", evidence=[f"worktree missing: {worktree}"])

    evidence: list[str] = []
    dirty = False
    ahead = 0
    behind = 0

    status_proc = _run_git(worktree, "status", "--short")
    if status_proc.returncode == 0:
        dirty = bool(status_proc.stdout.strip())
        if dirty:
            evidence.append("worktree has local changes")
    else:
        evidence.append("git status unavailable")

    counts_proc = _run_git(worktree, "rev-list", "--left-right", "--count", "origin/main...HEAD")
    if counts_proc.returncode == 0:
        parts = counts_proc.stdout.strip().split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            behind = int(parts[0])
            ahead = int(parts[1])
            if behind:
                evidence.append(f"branch behind origin/main by {behind} commits")
            if ahead:
                evidence.append(f"branch ahead of origin/main by {ahead} commits")
    else:
        evidence.append("ahead/behind unavailable")

    state = "clean"
    if behind > 0:
        state = "drifted"
    if dirty:
        state = "dirty" if state == "clean" else "dirty+drifted"
    return WorktreeStatus(state=state, dirty=dirty, ahead=ahead, behind=behind, evidence=evidence)


def _gh_json(*args: str) -> tuple[Any | None, str]:
    try:
        proc = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)

    if proc.returncode not in (0, 8):
        return None, proc.stderr.strip() or proc.stdout.strip() or "gh command failed"
    if not proc.stdout.strip():
        return None, "empty gh response"
    try:
        return json.loads(proc.stdout), ""
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _fetch_pr(branch: str) -> tuple[dict[str, Any] | None, str]:
    payload, error = _gh_json(
        "pr",
        "list",
        "--head",
        branch,
        "--state",
        "open",
        "--limit",
        "1",
        "--json",
        "number,headRefName,isDraft,url,title",
    )
    if payload is None:
        return None, error
    if not isinstance(payload, list) or not payload:
        return None, ""
    first = payload[0]
    return first if isinstance(first, dict) else None, ""


def _aggregate_checks_bucket(checks: list[dict[str, Any]]) -> str:
    buckets = {str(check.get("bucket", "")) for check in checks}
    if not buckets:
        return "unknown"
    if buckets & {"fail", "cancel"}:
        return "fail"
    if "pending" in buckets:
        return "pending"
    if buckets <= {"pass", "skipping"}:
        return "pass"
    return "unknown"


def _fetch_pr_checks(number: int) -> tuple[tuple[str, list[dict[str, str]]], str]:
    payload, error = _gh_json(
        "pr",
        "checks",
        str(number),
        "--json",
        "bucket,name,state,link,workflow",
    )
    if payload is None:
        return ("unknown", []), error
    if not isinstance(payload, list):
        return ("unknown", []), "unexpected gh pr checks payload"

    checks: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "bucket": str(item.get("bucket", "")),
                "name": str(item.get("name", "")),
                "state": str(item.get("state", "")),
                "workflow": str(item.get("workflow", "")),
            }
        )
    return (_aggregate_checks_bucket(checks), checks), ""


def _load_pr_truth(
    records: list[agent_bridge.LaneRecord],
) -> tuple[dict[str, PRTruth], list[str]]:
    truth: dict[str, PRTruth] = {}
    warnings: list[str] = []
    gh_unavailable = False

    for record in records:
        if gh_unavailable:
            break
        branch = record.branch.strip()
        if not branch or branch in truth:
            continue

        pr_payload, pr_error = _fetch_pr(branch)
        if pr_payload is None:
            if pr_error:
                warnings.append(f"{branch}: {pr_error}")
                gh_unavailable = True
            continue

        number = pr_payload.get("number")
        if not isinstance(number, int):
            warnings.append(f"{branch}: PR payload missing number")
            continue

        (checks_bucket, checks), checks_error = _fetch_pr_checks(number)
        truth[branch] = PRTruth(
            branch=branch,
            number=number,
            url=str(pr_payload.get("url", "")),
            is_draft=bool(pr_payload.get("isDraft", False)),
            checks_bucket=checks_bucket,
            checks=checks,
            error=checks_error,
        )
        if checks_error:
            warnings.append(f"{branch}: {checks_error}")
            gh_unavailable = True

    return truth, warnings


def _synthetic_lane_records(sessions: list[agent_bridge.Session]) -> list[agent_bridge.LaneRecord]:
    records: list[agent_bridge.LaneRecord] = []
    for session in sessions:
        records.append(
            agent_bridge.LaneRecord(
                lane_id=session.name,
                owner_session=session.name,
                status=session.status or "unknown",
                branch=session.branch,
                worktree=session.worktree,
                updated_at=_now_iso(),
            )
        )
    return records


def _decide_lane(
    record: agent_bridge.LaneRecord,
    session: agent_bridge.Session | None,
    session_text: str,
    worktree_status: WorktreeStatus,
    pr_truth: PRTruth | None,
) -> LaneDecision:
    evidence = list(worktree_status.evidence)
    session_signal = _classify_session_signal(session_text)
    if session and session.summary:
        evidence.append(f"session summary: {session.summary}")
    if record.conflict_reason:
        evidence.append(record.conflict_reason)
    if pr_truth and pr_truth.error:
        evidence.append(f"pr truth degraded: {pr_truth.error}")
    if pr_truth and pr_truth.checks_bucket:
        evidence.append(f"pr checks: {pr_truth.checks_bucket}")

    branch = record.branch or (session.branch if session else "")
    worktree = record.worktree or (session.worktree if session else "")

    if record.status == "conflict" or record.conflict_reason:
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="blocked",
            reason="lane ownership is ambiguous and needs human review",
            source=record.source,
            branch=branch,
            worktree=worktree,
            pr_number=pr_truth.number if pr_truth else None,
            pr_url=pr_truth.url if pr_truth else "",
            pr_checks_bucket=pr_truth.checks_bucket if pr_truth else "",
            evidence=evidence,
        )

    if session is None:
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="restart_from_main",
            reason="lane owner session is missing from the live session registry",
            source=record.source,
            branch=branch,
            worktree=worktree,
            evidence=evidence,
        )

    if session.status != "alive":
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="restart_from_main",
            reason=f"lane owner session is {session.status}",
            source=record.source,
            branch=branch,
            worktree=worktree,
            evidence=evidence,
        )

    if worktree_status.state == "missing":
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="restart_from_main",
            reason="lane worktree is missing and should be recreated from main",
            source=record.source,
            branch=branch,
            worktree=worktree,
            evidence=evidence,
        )

    if session_signal == "approve_prompt":
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="approve_prompt",
            reason="session output shows an explicit permission prompt",
            source=record.source,
            branch=branch,
            worktree=worktree,
            pr_number=pr_truth.number if pr_truth else None,
            pr_url=pr_truth.url if pr_truth else "",
            pr_checks_bucket=pr_truth.checks_bucket if pr_truth else "",
            evidence=evidence,
        )

    if pr_truth is not None:
        if pr_truth.error:
            reason = "pull request exists but live CI truth is unavailable"
            if session_signal == "wait_for_ci":
                reason = "session reports waiting for CI but live CI truth is unavailable"
                action = "blocked"
            else:
                action = "send_followup"
            return LaneDecision(
                lane_id=record.lane_id,
                owner_session=record.owner_session,
                status=record.status,
                next_action=action,
                reason=reason,
                source=record.source,
                branch=branch,
                worktree=worktree,
                pr_number=pr_truth.number,
                pr_url=pr_truth.url,
                pr_checks_bucket=pr_truth.checks_bucket,
                evidence=evidence,
            )

        if pr_truth.checks_bucket == "pending":
            return LaneDecision(
                lane_id=record.lane_id,
                owner_session=record.owner_session,
                status=record.status,
                next_action="wait_for_ci",
                reason="pull request checks are still pending",
                source=record.source,
                branch=branch,
                worktree=worktree,
                pr_number=pr_truth.number,
                pr_url=pr_truth.url,
                pr_checks_bucket=pr_truth.checks_bucket,
                evidence=evidence,
            )

        if pr_truth.checks_bucket == "fail":
            return LaneDecision(
                lane_id=record.lane_id,
                owner_session=record.owner_session,
                status=record.status,
                next_action="send_followup",
                reason="pull request checks are failing and the owner needs a bounded fix prompt",
                source=record.source,
                branch=branch,
                worktree=worktree,
                pr_number=pr_truth.number,
                pr_url=pr_truth.url,
                pr_checks_bucket=pr_truth.checks_bucket,
                evidence=evidence,
            )

        if pr_truth.checks_bucket == "pass":
            if pr_truth.is_draft:
                return LaneDecision(
                    lane_id=record.lane_id,
                    owner_session=record.owner_session,
                    status=record.status,
                    next_action="send_followup",
                    reason="draft pull request is green but still needs the owner to mark it ready",
                    source=record.source,
                    branch=branch,
                    worktree=worktree,
                    pr_number=pr_truth.number,
                    pr_url=pr_truth.url,
                    pr_checks_bucket=pr_truth.checks_bucket,
                    evidence=evidence,
                )
            return LaneDecision(
                lane_id=record.lane_id,
                owner_session=record.owner_session,
                status=record.status,
                next_action="ready_for_review",
                reason="pull request checks passed and the lane is ready for review",
                source=record.source,
                branch=branch,
                worktree=worktree,
                pr_number=pr_truth.number,
                pr_url=pr_truth.url,
                pr_checks_bucket=pr_truth.checks_bucket,
                evidence=evidence,
            )

    if session_signal == "blocked":
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="blocked",
            reason="session summary reports a blocked state without enough PR truth to advance safely",
            source=record.source,
            branch=branch,
            worktree=worktree,
            evidence=evidence,
        )

    if session_signal == "wait_for_ci":
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="blocked",
            reason="session reports waiting for CI but no open pull request was found for the lane branch",
            source=record.source,
            branch=branch,
            worktree=worktree,
            evidence=evidence,
        )

    if worktree_status.behind > 0 and not branch:
        return LaneDecision(
            lane_id=record.lane_id,
            owner_session=record.owner_session,
            status=record.status,
            next_action="restart_from_main",
            reason="lane has drifted from main and has no tracked branch to reconcile",
            source=record.source,
            worktree=worktree,
            evidence=evidence,
        )

    return LaneDecision(
        lane_id=record.lane_id,
        owner_session=record.owner_session,
        status=record.status,
        next_action="send_followup",
        reason="no terminal blocker or PR truth was found; send one bounded follow-up prompt",
        source=record.source,
        branch=branch,
        worktree=worktree,
        evidence=evidence,
    )


def collect_supervisor_snapshot() -> SupervisorSnapshot:
    sessions = agent_bridge.discover()
    agent_bridge._enrich_prs(sessions)  # noqa: SLF001
    records = agent_bridge._sync_lane_records(  # noqa: SLF001
        agent_bridge._load_lane_registry(),  # noqa: SLF001
        sessions,
    )
    if not records:
        records = _synthetic_lane_records(sessions)

    pr_truth_by_branch, warnings = _load_pr_truth(records)
    session_map = {session.name: session for session in sessions}
    decisions: list[LaneDecision] = []

    for record in records:
        session = session_map.get(record.owner_session)
        branch = record.branch or (session.branch if session else "")
        decisions.append(
            _decide_lane(
                record,
                session,
                _session_text(session),
                _inspect_worktree(record.worktree or (session.worktree if session else "")),
                pr_truth_by_branch.get(branch) if branch else None,
            )
        )

    decisions.sort(key=lambda decision: (decision.next_action, decision.lane_id))
    return SupervisorSnapshot(generated_at=_now_iso(), decisions=decisions, warnings=warnings)


def _render_text(snapshot: SupervisorSnapshot) -> str:
    if not snapshot.decisions:
        return "No lanes or sessions to supervise."

    lines = [
        f"Supervisor snapshot @ {snapshot.generated_at}",
        f"{'LANE':<22} {'OWNER':<24} {'ACTION':<18} {'PR':<8} REASON",
        "-" * 120,
    ]
    for decision in snapshot.decisions:
        pr = f"#{decision.pr_number}" if decision.pr_number else "-"
        lines.append(
            f"{decision.lane_id[:22]:<22} {decision.owner_session[:24]:<24} "
            f"{decision.next_action[:18]:<18} {pr:<8} {decision.reason}"
        )
    if snapshot.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in snapshot.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Passive supervision loop for agent bridge lanes.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one supervision cycle and exit. This is the default when no interval is set.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Poll continuously every N seconds.",
    )
    parser.add_argument("--json", action="store_true", help="Render machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_once = args.once or args.interval is None

    while True:
        snapshot = collect_supervisor_snapshot()
        if args.json:
            print(json.dumps(snapshot.to_dict(), indent=2))
        else:
            print(_render_text(snapshot))
        if run_once:
            return 0
        time.sleep(max(args.interval or 0.0, 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
