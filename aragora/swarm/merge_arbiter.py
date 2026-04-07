"""Admin merge arbiter for the boss loop.

Polls open PRs matching configured branch prefixes and auto-merges them
when all 5 required CI checks pass.  Designed to run alongside the boss
loop for unattended overnight operation.

Usage:
    arbiter = MergeArbiter(MergeArbiterConfig(branch_prefixes=["boss-harvest"]))
    await arbiter.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

REQUIRED_CHECKS: list[str] = [
    "lint",
    "typecheck",
    "sdk-parity",
    "Generate & Validate",
    "TypeScript SDK Type Check",
]


@dataclass
class MergeArbiterConfig:
    """Configuration for the merge arbiter polling loop."""

    repo: str = "synaptent/aragora"
    branch_prefixes: list[str] = field(default_factory=lambda: ["boss-harvest", "codex/"])
    poll_interval_seconds: float = 120.0
    max_runtime_hours: float = 12.0
    max_consecutive_failures: int = 3
    dry_run: bool = False


@dataclass
class MergeResult:
    """Outcome of a single merge attempt."""

    pr_number: int
    branch: str
    success: bool
    reason: str


@dataclass
class ArbiterSummary:
    """Final summary when the arbiter exits."""

    merged: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    polls: int = 0
    stop_reason: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "merged": self.merged,
            "skipped": self.skipped,
            "failed": self.failed,
            "polls": self.polls,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


def _classify_required_checks(
    checks: dict[str, str],
    *,
    required_checks: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split required checks into missing and failing buckets."""
    missing: list[str] = []
    failing: list[str] = []
    for name in required_checks or REQUIRED_CHECKS:
        status = checks.get(name)
        if status is None:
            missing.append(name)
        elif status != "SUCCESS":
            failing.append(f"{name}={status}")
    return missing, failing


def _run_gh(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command and return the result."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _list_candidate_prs(config: MergeArbiterConfig) -> list[dict]:
    """Return open PRs whose head branch matches any configured prefix."""
    result = _run_gh(
        [
            "pr",
            "list",
            "--repo",
            config.repo,
            "--state",
            "open",
            "--json",
            "number,headRefName,isDraft",
            "--limit",
            "100",
        ]
    )
    if result.returncode != 0:
        logger.warning("gh pr list failed: %s", result.stderr.strip())
        return []
    try:
        prs = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse gh pr list output")
        return []
    candidates = []
    for pr in prs:
        branch = pr.get("headRefName", "")
        if any(branch.startswith(prefix) for prefix in config.branch_prefixes):
            candidates.append(pr)
    return candidates


def _get_check_status(pr_number: int, repo: str) -> dict[str, str]:
    """Return a mapping of check-name -> conclusion for a PR.

    Merges both status checks and GitHub Actions check runs.
    """
    result = _run_gh(
        [
            "pr",
            "checks",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "name,state",
        ]
    )
    if result.returncode != 0:
        logger.debug("gh pr checks failed for #%d: %s", pr_number, result.stderr.strip())
        return {}
    try:
        checks = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return {}
    return {c["name"]: c.get("state", "").upper() for c in checks if "name" in c}


def _promote_draft(pr_number: int, repo: str) -> bool:
    """Convert a draft PR to ready-for-review."""
    result = _run_gh(
        [
            "pr",
            "ready",
            str(pr_number),
            "--repo",
            repo,
        ]
    )
    if result.returncode != 0:
        logger.warning("Failed to promote draft PR #%d: %s", pr_number, result.stderr.strip())
        return False
    logger.info("Promoted draft PR #%d to ready", pr_number)
    return True


def _merge_pr(pr_number: int, repo: str) -> tuple[bool, str]:
    """Squash-merge a PR with admin override.  Returns (success, reason)."""
    result = _run_gh(
        [
            "pr",
            "merge",
            str(pr_number),
            "--repo",
            repo,
            "--admin",
            "--squash",
            "--delete-branch",
        ]
    )
    if result.returncode != 0:
        reason = result.stderr.strip() or "unknown error"
        return False, reason
    return True, "merged"


def _evaluate_pr(pr: dict, config: MergeArbiterConfig) -> MergeResult:
    """Evaluate a single PR and merge it if all required checks pass."""
    pr_number: int = pr["number"]
    branch: str = pr.get("headRefName", "")
    is_draft: bool = pr.get("isDraft", False)

    if is_draft:
        if not _promote_draft(pr_number, config.repo):
            return MergeResult(pr_number, branch, False, "failed to promote draft")
        # After promoting, checks may not have run yet -- skip this cycle
        return MergeResult(pr_number, branch, False, "promoted from draft, waiting for checks")

    checks = _get_check_status(pr_number, config.repo)
    if not checks:
        return MergeResult(pr_number, branch, False, "no checks found")

    missing, failing = _classify_required_checks(checks)

    if missing:
        return MergeResult(pr_number, branch, False, f"missing checks: {', '.join(missing)}")
    if failing:
        return MergeResult(pr_number, branch, False, f"failing checks: {', '.join(failing)}")

    # All required checks pass
    if config.dry_run:
        logger.info("[dry-run] Would merge PR #%d (%s)", pr_number, branch)
        return MergeResult(pr_number, branch, True, "dry-run: would merge")

    ok, reason = _merge_pr(pr_number, config.repo)
    if ok:
        logger.info("Merged PR #%d (%s)", pr_number, branch)
    else:
        logger.warning("Failed to merge PR #%d: %s", pr_number, reason)
    return MergeResult(pr_number, branch, ok, reason)


class MergeArbiter:
    """Polling loop that auto-merges boss-loop PRs when CI passes."""

    def __init__(self, config: MergeArbiterConfig | None = None) -> None:
        self.config = config or MergeArbiterConfig()
        self._consecutive_failures = 0

    async def run(self) -> ArbiterSummary:
        """Run the polling loop until max runtime or circuit breaker trips."""
        summary = ArbiterSummary()
        start = time.monotonic()
        deadline = start + self.config.max_runtime_hours * 3600

        logger.info(
            "Merge arbiter started: repo=%s prefixes=%s interval=%ds max_hours=%.1f dry_run=%s",
            self.config.repo,
            self.config.branch_prefixes,
            int(self.config.poll_interval_seconds),
            self.config.max_runtime_hours,
            self.config.dry_run,
        )

        while time.monotonic() < deadline:
            summary.polls += 1
            any_failure_this_poll = False
            any_merge_this_poll = False

            candidates = _list_candidate_prs(self.config)
            logger.debug("Poll %d: %d candidate PRs", summary.polls, len(candidates))

            for pr in candidates:
                result = _evaluate_pr(pr, self.config)
                if result.success:
                    summary.merged.append(result.pr_number)
                    any_merge_this_poll = True
                    logger.info(
                        "PR #%d: %s (%s)",
                        result.pr_number,
                        result.reason,
                        result.branch,
                    )
                elif "failing" in result.reason or "failed" in result.reason:
                    summary.failed.append(result.pr_number)
                    any_failure_this_poll = True
                    logger.info(
                        "PR #%d skipped: %s (%s)",
                        result.pr_number,
                        result.reason,
                        result.branch,
                    )
                else:
                    summary.skipped.append(result.pr_number)
                    logger.debug(
                        "PR #%d waiting: %s (%s)",
                        result.pr_number,
                        result.reason,
                        result.branch,
                    )

            # Circuit breaker: track consecutive polls with only failures
            if any_failure_this_poll and not any_merge_this_poll:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

            if self._consecutive_failures >= self.config.max_consecutive_failures:
                summary.stop_reason = (
                    f"circuit breaker: {self._consecutive_failures} consecutive failure-only polls"
                )
                logger.warning("Merge arbiter stopping: %s", summary.stop_reason)
                break

            await asyncio.sleep(self.config.poll_interval_seconds)
        else:
            summary.stop_reason = "max runtime reached"

        summary.elapsed_seconds = time.monotonic() - start
        logger.info(
            "Merge arbiter finished: %s (merged=%d skipped=%d failed=%d polls=%d)",
            summary.stop_reason,
            len(summary.merged),
            len(summary.skipped),
            len(summary.failed),
            summary.polls,
        )
        return summary
