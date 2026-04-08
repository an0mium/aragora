"""Admin merge arbiter for automation PRs.

Polls open automation PRs matching configured branch prefixes and auto-merges
only ready PRs whose branch-protection checks and ready-only full-suite checks
have all passed. Draft PRs are never auto-merged here; the boss loop owns draft
promotion separately.

Usage:
    arbiter = MergeArbiter()
    await arbiter.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

REQUIRED_CHECKS: list[str] = [
    "lint",
    "typecheck",
    "sdk-parity",
    "Generate & Validate",
    "TypeScript SDK Type Check",
]
AUTOMATION_BRANCH_PREFIXES: list[str] = [
    "codex/",
    "factory/",
    "aragora/boss-harvest/",
]
PASSING_CHECK_STATES = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})
READY_SUITE_GATE_CHECKS = frozenset({"Prioritize Required Checks"})
REDUCED_LANE_ONLY_CHECKS = frozenset(
    {
        "PR Admission Signal (Advisory)",
        "Prioritize Required Checks",
        "OpenAPI Scope",
        "SDK Change Detection",
        "publish-draft-pr",
        "review",
        "changes",
    }
)


@dataclass
class MergeArbiterConfig:
    """Configuration for the merge arbiter polling loop."""

    repo: str = "synaptent/aragora"
    branch_prefixes: list[str] = field(default_factory=lambda: list(AUTOMATION_BRANCH_PREFIXES))
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


def _normalize_branch_prefixes(branch_prefixes: list[str] | None) -> list[str]:
    """Normalize configured prefixes to the canonical automation branch forms."""
    raw_prefixes = list(branch_prefixes or AUTOMATION_BRANCH_PREFIXES)
    normalized: list[str] = []
    seen: set[str] = set()
    aliases = {
        "boss-harvest": "aragora/boss-harvest/",
        "boss-harvest/": "aragora/boss-harvest/",
        "aragora/boss-harvest": "aragora/boss-harvest/",
        "codex": "codex/",
        "factory": "factory/",
    }
    for prefix in raw_prefixes:
        value = aliases.get(str(prefix or "").strip(), str(prefix or "").strip())
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized or list(AUTOMATION_BRANCH_PREFIXES)


@lru_cache(maxsize=8)
def _get_required_checks(repo: str, base_branch: str = "main") -> list[str]:
    """Load required branch-protection contexts, with a local fallback."""
    result = _run_gh(
        [
            "api",
            f"repos/{repo}/branches/{base_branch}/protection",
            "--jq",
            ".required_status_checks.contexts",
        ]
    )
    if result.returncode != 0:
        return list(REQUIRED_CHECKS)
    try:
        contexts = json.loads(result.stdout or "[]")
    except (json.JSONDecodeError, TypeError):
        return list(REQUIRED_CHECKS)
    if not isinstance(contexts, list):
        return list(REQUIRED_CHECKS)
    normalized = [str(item).strip() for item in contexts if str(item).strip()]
    return normalized or list(REQUIRED_CHECKS)


def _classify_non_passing_checks(
    checks: dict[str, str],
    *,
    ignored_checks: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split non-passing checks into waiting and failing buckets."""
    waiting: list[str] = []
    failing: list[str] = []
    ignored = ignored_checks or set()
    for name in sorted(checks):
        if name in ignored:
            continue
        status = checks.get(name, "")
        if status in PASSING_CHECK_STATES:
            continue
        detail = f"{name}={status}"
        if status in {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED", "WAITING", "REQUESTED"}:
            waiting.append(detail)
        else:
            failing.append(detail)
    return waiting, failing


def _ready_suite_check_names(
    checks: dict[str, str],
    *,
    required_checks: list[str],
) -> list[str]:
    ignored = set(required_checks) | set(REDUCED_LANE_ONLY_CHECKS)
    return sorted(name for name in checks if name not in ignored)


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
    prefixes = _normalize_branch_prefixes(config.branch_prefixes)
    result = _run_gh(
        [
            "pr",
            "list",
            "--repo",
            config.repo,
            "--state",
            "open",
            "--json",
            "number,headRefName,headRefOid,isDraft",
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
        if any(branch.startswith(prefix) for prefix in prefixes):
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
    """Mark a draft PR as ready for review."""
    result = _run_gh(["pr", "ready", str(pr_number), "--repo", repo], timeout=30.0)
    return result.returncode == 0


def _merge_pr(
    pr_number: int,
    repo: str,
    head_sha: str | None = None,
) -> tuple[bool, str]:
    """Squash-merge a PR with admin override.  Returns (success, reason)."""
    args = [
        "pr",
        "merge",
        str(pr_number),
        "--repo",
        repo,
        "--admin",
        "--squash",
        "--delete-branch",
    ]
    if head_sha:
        args.extend(["--match-head-commit", head_sha])
    result = _run_gh(args)
    if result.returncode != 0:
        reason = result.stderr.strip() or "unknown error"
        return False, reason
    return True, "merged"


def _evaluate_pr(pr: dict, config: MergeArbiterConfig) -> MergeResult:
    """Evaluate a single PR and merge it if all required checks pass."""
    pr_number: int = pr["number"]
    branch: str = pr.get("headRefName", "")
    head_sha = pr.get("headRefOid")
    is_draft: bool = pr.get("isDraft", False)
    required_checks = _get_required_checks(config.repo)

    if is_draft:
        checks = _get_check_status(pr_number, config.repo)
        if not checks:
            return MergeResult(
                pr_number,
                branch,
                False,
                "draft PR: never auto-merged; no fast required checks reported yet",
            )
        missing, failing = _classify_required_checks(checks, required_checks=required_checks)
        if missing or failing:
            reason_parts = ["draft PR: never auto-merged"]
            if missing:
                reason_parts.append(f"fast required checks missing: {', '.join(missing)}")
            if failing:
                reason_parts.append(f"fast required checks failing: {', '.join(failing)}")
            return MergeResult(pr_number, branch, False, "; ".join(reason_parts))
        return MergeResult(
            pr_number,
            branch,
            False,
            "draft PR: fast required checks passed; waiting for boss-loop promotion to ready",
        )

    checks = _get_check_status(pr_number, config.repo)
    if not checks:
        return MergeResult(pr_number, branch, False, "no checks found")

    missing, failing = _classify_required_checks(checks, required_checks=required_checks)

    if missing:
        return MergeResult(
            pr_number,
            branch,
            False,
            f"missing required checks: {', '.join(missing)}",
        )
    if failing:
        return MergeResult(
            pr_number,
            branch,
            False,
            f"failing required checks: {', '.join(failing)}",
        )

    missing_ready_gates = sorted(name for name in READY_SUITE_GATE_CHECKS if name not in checks)
    if missing_ready_gates:
        return MergeResult(
            pr_number,
            branch,
            False,
            f"ready PR missing full-suite gate checks: {', '.join(missing_ready_gates)}",
        )

    ready_suite_checks = _ready_suite_check_names(checks, required_checks=required_checks)
    if not ready_suite_checks:
        return MergeResult(
            pr_number,
            branch,
            False,
            "ready PR still only has reduced fast-lane checks; no full-suite checks reported yet",
        )

    ready_suite_statuses = {name: checks[name] for name in ready_suite_checks}
    waiting_ready, failing_ready = _classify_non_passing_checks(ready_suite_statuses)
    if waiting_ready:
        return MergeResult(
            pr_number,
            branch,
            False,
            f"waiting on full-suite checks: {', '.join(waiting_ready)}",
        )
    if failing_ready:
        return MergeResult(
            pr_number,
            branch,
            False,
            f"failing full-suite checks: {', '.join(failing_ready)}",
        )

    if config.dry_run:
        logger.info("[dry-run] Would merge PR #%d (%s)", pr_number, branch)
        return MergeResult(pr_number, branch, True, "dry-run: would merge")

    ok, reason = _merge_pr(pr_number, config.repo, head_sha)
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
