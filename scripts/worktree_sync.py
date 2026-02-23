#!/usr/bin/env python3
"""Worktree Sync Coordinator.

Keeps parallel worktrees in sync with main and detects file overlap
conflicts before they become git conflicts.

Usage:
    python scripts/worktree_sync.py                # Full sync report
    python scripts/worktree_sync.py --rebase       # Rebase all worktrees onto main
    python scripts/worktree_sync.py --conflicts     # Only check for file overlaps
    python scripts/worktree_sync.py --json          # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class WorktreeStatus:
    """Status of a single worktree."""

    branch: str
    path: str
    track: str
    ahead: int = 0
    behind: int = 0
    changed_files: list[str] = field(default_factory=list)
    status: str = "ok"  # ok | behind | diverged | detached
    last_commit_time: str = ""
    last_commit_msg: str = ""


@dataclass
class FileOverlap:
    """Files modified by multiple worktrees."""

    file_path: str
    branches: list[str] = field(default_factory=list)
    severity: str = "low"  # low | medium | high


@dataclass
class SyncReport:
    """Full sync reconciliation report."""

    timestamp: str
    base_branch: str
    worktrees: list[WorktreeStatus] = field(default_factory=list)
    overlaps: list[FileOverlap] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a git command and return result."""
    cmd = ["git"] + list(args)
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_worktree_list() -> list[tuple[str, str]]:
    """Parse `git worktree list --porcelain` into (path, branch) pairs."""
    result = run_git("worktree", "list", "--porcelain")
    if result.returncode != 0:
        return []

    worktrees: list[tuple[str, str]] = []
    current_path: str | None = None
    current_branch: str | None = None

    for line in result.stdout.split("\n"):
        line = line.strip()
        if line.startswith("worktree "):
            current_path = line[len("worktree ") :]
            current_branch = None
        elif line.startswith("branch refs/heads/"):
            current_branch = line[len("branch refs/heads/") :]
        elif line == "" and current_path and current_branch:
            # Skip the main worktree (the repo root itself)
            if current_branch not in ("main", "master"):
                worktrees.append((current_path, current_branch))
            current_path = None
            current_branch = None

    # Handle last entry
    if current_path and current_branch and current_branch not in ("main", "master"):
        worktrees.append((current_path, current_branch))

    return worktrees


def get_worktree_status(path: str, branch: str, base_branch: str) -> WorktreeStatus:
    """Get detailed status for a worktree."""
    # Extract track name from branch
    track = branch
    for prefix in ("dev/", "work/", "sprint/"):
        if track.startswith(prefix):
            track = track[len(prefix) :]
            break
    # Strip timestamp suffixes
    parts = track.rsplit("-", 1)
    if parts[-1].isdigit() and len(parts[-1]) >= 6:
        track = parts[0]

    status = WorktreeStatus(
        branch=branch,
        path=path,
        track=track,
    )

    # Get ahead/behind counts
    result = run_git(
        "rev-list",
        "--left-right",
        "--count",
        f"{base_branch}...{branch}",
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split()
        if len(parts) == 2:
            status.behind = int(parts[0])
            status.ahead = int(parts[1])

    # Get changed files relative to base
    result = run_git("diff", "--name-only", f"{base_branch}...{branch}")
    if result.returncode == 0:
        status.changed_files = [f for f in result.stdout.strip().split("\n") if f]

    # Get last commit info
    result = run_git(
        "log",
        "-1",
        "--format=%aI|%s",
        branch,
    )
    if result.returncode == 0 and "|" in result.stdout:
        parts = result.stdout.strip().split("|", 1)
        status.last_commit_time = parts[0]
        status.last_commit_msg = parts[1][:80]

    # Determine status
    if status.behind > 0 and status.ahead > 0:
        status.status = "diverged"
    elif status.behind > 0:
        status.status = "behind"
    else:
        status.status = "ok"

    return status


def detect_file_overlaps(worktrees: list[WorktreeStatus]) -> list[FileOverlap]:
    """Detect files modified by multiple worktrees."""
    file_to_branches: dict[str, list[str]] = {}

    for wt in worktrees:
        for f in wt.changed_files:
            if f not in file_to_branches:
                file_to_branches[f] = []
            file_to_branches[f].append(wt.branch)

    overlaps: list[FileOverlap] = []
    for file_path, branches in file_to_branches.items():
        if len(branches) > 1:
            # Determine severity based on overlap count and file type
            if len(branches) > 3:
                severity = "high"
            elif len(branches) > 2 or file_path.endswith((".py", ".ts", ".tsx")):
                severity = "medium"
            else:
                severity = "low"

            overlaps.append(
                FileOverlap(
                    file_path=file_path,
                    branches=branches,
                    severity=severity,
                )
            )

    # Sort by severity (high first)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    overlaps.sort(key=lambda o: severity_order.get(o.severity, 3))
    return overlaps


def generate_recommendations(
    worktrees: list[WorktreeStatus],
    overlaps: list[FileOverlap],
) -> list[str]:
    """Generate actionable recommendations from the sync report."""
    recs: list[str] = []

    # Check for stale worktrees
    behind_branches = [wt for wt in worktrees if wt.behind > 5]
    if behind_branches:
        names = ", ".join(wt.branch for wt in behind_branches)
        recs.append(f"Rebase stale worktrees: {names} (5+ commits behind main)")

    # Check for high-severity overlaps
    high_overlaps = [o for o in overlaps if o.severity == "high"]
    if high_overlaps:
        files = ", ".join(o.file_path for o in high_overlaps[:3])
        recs.append(
            f"URGENT: High-severity file overlaps detected in {files}. "
            f"Coordinate with affected tracks before merging."
        )

    # Check for diverged branches
    diverged = [wt for wt in worktrees if wt.status == "diverged"]
    if diverged:
        names = ", ".join(wt.branch for wt in diverged)
        recs.append(f"Diverged branches need rebase: {names}")

    # Suggest merge order (least risky first)
    ready = [wt for wt in worktrees if wt.ahead > 0 and wt.status == "ok"]
    if ready:
        ready.sort(key=lambda wt: len(wt.changed_files))
        names = ", ".join(wt.branch for wt in ready[:3])
        recs.append(f"Suggested merge order (fewest changes first): {names}")

    if not recs:
        recs.append("All worktrees are in good shape. No action needed.")

    return recs


def rebase_worktree(path: str, branch: str, base_branch: str) -> bool:
    """Rebase a worktree branch onto the latest base branch."""
    wt_path = Path(path)

    # Fetch latest
    run_git("fetch", "origin", base_branch, cwd=wt_path)

    # Rebase
    result = run_git("rebase", base_branch, cwd=wt_path)
    if result.returncode != 0:
        # Abort failed rebase
        run_git("rebase", "--abort", cwd=wt_path)
        return False
    return True


def build_report(base_branch: str = "main") -> SyncReport:
    """Build a full sync report."""
    report = SyncReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        base_branch=base_branch,
    )

    # Enumerate worktrees
    worktree_list = parse_worktree_list()
    if not worktree_list:
        report.errors.append("No worktrees found.")
        return report

    # Get status for each
    for path, branch in worktree_list:
        try:
            status = get_worktree_status(path, branch, base_branch)
            report.worktrees.append(status)
        except Exception as e:
            report.errors.append(f"Error checking {branch}: {e}")

    # Detect overlaps
    report.overlaps = detect_file_overlaps(report.worktrees)

    # Generate recommendations
    report.recommendations = generate_recommendations(report.worktrees, report.overlaps)

    return report


def print_report(report: SyncReport) -> None:
    """Print human-readable sync report."""
    print("\n" + "=" * 60)
    print("  Worktree Sync Report")
    print(f"  {report.timestamp}")
    print("=" * 60)

    if not report.worktrees:
        print("\n  No worktrees found.\n")
        return

    # Worktree status table
    print(f"\n  {'Branch':<40} {'Ahead':>6} {'Behind':>7} {'Files':>6} {'Status':>10}")
    print("  " + "-" * 72)
    for wt in report.worktrees:
        status_color = {
            "ok": "\033[32m",  # green
            "behind": "\033[33m",  # yellow
            "diverged": "\033[31m",  # red
        }.get(wt.status, "")
        reset = "\033[0m"
        print(
            f"  {wt.branch:<40} {wt.ahead:>6} {wt.behind:>7} "
            f"{len(wt.changed_files):>6} "
            f"{status_color}{wt.status:>10}{reset}"
        )

    # File overlaps
    if report.overlaps:
        print(f"\n  File Overlaps ({len(report.overlaps)} files shared between branches):")
        print("  " + "-" * 72)
        for overlap in report.overlaps:
            sev_color = {
                "high": "\033[31m",
                "medium": "\033[33m",
                "low": "\033[34m",
            }.get(overlap.severity, "")
            reset = "\033[0m"
            branches = ", ".join(overlap.branches)
            print(f"  {sev_color}[{overlap.severity:>6}]{reset} {overlap.file_path}")
            print(f"           -> {branches}")

    # Recommendations
    print("\n  Recommendations:")
    print("  " + "-" * 72)
    for rec in report.recommendations:
        print(f"  * {rec}")

    # Errors
    if report.errors:
        print("\n  Errors:")
        for err in report.errors:
            print(f"  ! {err}")

    print("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Worktree sync coordinator — keeps parallel worktrees in sync"
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="Rebase all behind/diverged worktrees onto main",
    )
    parser.add_argument(
        "--conflicts",
        action="store_true",
        help="Only check for file overlap conflicts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Base branch to compare against (default: main)",
    )
    args = parser.parse_args()

    report = build_report(base_branch=args.base_branch)

    # Rebase mode
    if args.rebase:
        for wt in report.worktrees:
            if wt.status in ("behind", "diverged"):
                print(f"Rebasing {wt.branch}...", end=" ")
                if rebase_worktree(wt.path, wt.branch, args.base_branch):
                    print("OK")
                    wt.status = "ok"
                    wt.behind = 0
                else:
                    print("FAILED (conflicts — manual rebase needed)")
        # Refresh report after rebases
        report = build_report(base_branch=args.base_branch)

    # Output
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    elif args.conflicts:
        if report.overlaps:
            for overlap in report.overlaps:
                branches = ", ".join(overlap.branches)
                print(f"[{overlap.severity}] {overlap.file_path} -> {branches}")
            return 1 if any(o.severity == "high" for o in report.overlaps) else 0
        else:
            print("No file overlaps detected.")
            return 0
    else:
        print_report(report)

    # Exit code: 1 if high-severity overlaps exist
    if any(o.severity == "high" for o in report.overlaps):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
