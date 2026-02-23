#!/usr/bin/env python3
"""CI-Aware Push Gate.

Prevents concurrent pushes to main that cancel each other's CI runs.
Checks GitHub Actions status before pushing, queues if CI is in-flight.

Usage:
    python scripts/ci_gate.py                    # Push main with CI gate
    python scripts/ci_gate.py --branch dev/core  # Push specific branch
    python scripts/ci_gate.py --status           # Just check CI status
    python scripts/ci_gate.py --wait             # Wait for CI to finish, then push
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum time to wait for CI to complete (minutes)
MAX_WAIT_MINUTES = 15
POLL_INTERVAL_SECONDS = 30


@dataclass
class CIStatus:
    """Current CI pipeline status."""

    is_running: bool
    run_id: int | None = None
    workflow_name: str = ""
    status: str = ""  # queued | in_progress | completed
    conclusion: str = ""  # success | failure | cancelled
    branch: str = ""
    url: str = ""
    started_at: str = ""
    elapsed_minutes: float = 0.0


def run_gh(*args: str) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    cmd = ["gh"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def run_git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command."""
    cmd = ["git"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def check_gh_available() -> bool:
    """Check if gh CLI is installed and authenticated."""
    result = run_gh("auth", "status")
    return result.returncode == 0


def get_ci_status(branch: str = "main") -> list[CIStatus]:
    """Check current CI run status for a branch."""
    result = run_gh(
        "run",
        "list",
        "--branch",
        branch,
        "--limit",
        "5",
        "--json",
        "databaseId,name,status,conclusion,headBranch,url,createdAt",
    )
    if result.returncode != 0:
        logger.warning("Failed to check CI status: %s", result.stderr)
        return []

    try:
        runs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    statuses: list[CIStatus] = []
    for run in runs:
        status = run.get("status", "")
        is_running = status in ("queued", "in_progress")
        statuses.append(
            CIStatus(
                is_running=is_running,
                run_id=run.get("databaseId"),
                workflow_name=run.get("name", ""),
                status=status,
                conclusion=run.get("conclusion", ""),
                branch=run.get("headBranch", ""),
                url=run.get("url", ""),
                started_at=run.get("createdAt", ""),
            )
        )

    return statuses


def any_ci_running(branch: str = "main") -> tuple[bool, list[CIStatus]]:
    """Check if any CI workflow is currently running on a branch."""
    statuses = get_ci_status(branch)
    running = [s for s in statuses if s.is_running]
    return len(running) > 0, running


def wait_for_ci(branch: str = "main", max_minutes: float = MAX_WAIT_MINUTES) -> bool:
    """Wait for CI to complete on a branch.

    Returns True if CI completed (or no CI was running).
    Returns False if timeout reached.
    """
    start = time.time()
    deadline = start + (max_minutes * 60)

    while time.time() < deadline:
        is_running, running = any_ci_running(branch)
        if not is_running:
            return True

        elapsed = (time.time() - start) / 60
        remaining = max_minutes - elapsed
        names = ", ".join(r.workflow_name for r in running)
        print(f"  CI in progress: {names} ({elapsed:.1f}m elapsed, {remaining:.1f}m remaining)")
        time.sleep(POLL_INTERVAL_SECONDS)

    return False


def push_with_gate(
    branch: str = "main",
    remote: str = "origin",
    wait: bool = False,
    force: bool = False,
) -> int:
    """Push to remote with CI gate.

    Returns exit code (0 = success).
    """
    print(f"CI Gate: Checking status for {branch}...")

    if not check_gh_available():
        print("WARNING: gh CLI not available — pushing without CI check.")
        result = run_git("push", remote, branch)
        return result.returncode

    # Check current CI status
    is_running, running = any_ci_running(branch)

    if is_running:
        names = ", ".join(f"{r.workflow_name} ({r.status})" for r in running)
        print(f"  CI is currently running: {names}")

        if wait:
            print(f"  Waiting up to {MAX_WAIT_MINUTES} minutes for CI to complete...")
            if not wait_for_ci(branch):
                print("  TIMEOUT: CI did not complete in time.")
                if not force:
                    print("  Aborting push. Use --force to push anyway.")
                    return 1
                print("  Force-pushing despite running CI.")
        elif not force:
            print("  Aborting push to avoid cancelling CI run.")
            print("  Options:")
            print("    --wait   Wait for CI to complete, then push")
            print("    --force  Push anyway (may cancel running CI)")
            return 1
        else:
            print("  Force-pushing despite running CI.")

    # Push
    print(f"  Pushing {branch} to {remote}...")
    result = run_git("push", remote, branch)

    if result.returncode != 0:
        print(f"  Push failed: {result.stderr.strip()}")
        return result.returncode

    print("  Push successful.")

    # Check that CI started
    time.sleep(5)
    new_statuses = get_ci_status(branch)
    new_running = [s for s in new_statuses if s.is_running]
    if new_running:
        for s in new_running:
            print(f"  CI started: {s.workflow_name} — {s.url}")
    else:
        print("  No CI workflow triggered (check workflow config).")

    return 0


def print_status(branch: str = "main") -> None:
    """Print current CI status."""
    if not check_gh_available():
        print("gh CLI not available. Install: https://cli.github.com")
        return

    statuses = get_ci_status(branch)
    if not statuses:
        print(f"No recent CI runs found for {branch}.")
        return

    print(f"\nCI Status for {branch}:")
    print(f"  {'Workflow':<30} {'Status':<15} {'Conclusion':<12} {'URL'}")
    print("  " + "-" * 80)
    for s in statuses:
        status_icon = {
            "queued": "🟡",
            "in_progress": "🔵",
            "completed": "🟢" if s.conclusion == "success" else "🔴",
        }.get(s.status, "⚪")
        print(f"  {s.workflow_name:<30} {status_icon} {s.status:<12} {s.conclusion:<12} {s.url}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="CI-aware push gate — prevents push conflicts")
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to push (default: main)",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Remote to push to (default: origin)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Just check CI status, don't push",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for running CI to complete before pushing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Push even if CI is running",
    )
    args = parser.parse_args()

    if args.status:
        print_status(args.branch)
        return 0

    return push_with_gate(
        branch=args.branch,
        remote=args.remote,
        wait=args.wait,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
